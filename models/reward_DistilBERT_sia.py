import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

SAVE_PATH = "./checkpoints/reward/BERT_sia"
MODEL_NAME = "distilbert-base-uncased"

# ──────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────
def tokenize_ultrafeedback_BERT(example, *, tokenizer, max_length: int = 512):
    prompt = example["prompt"]
    chosen = example["chosen"][1]["content"]
    rejected = example["rejected"][1]["content"]

    base = "system\nYou are a helpful assistant.\nuser\n" + prompt.strip() + "\nassistant\n"

    chosen_text = base + chosen.strip()
    rejected_text = base + rejected.strip()

    chosen_enc = tokenizer(
        chosen_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    rejected_enc = tokenizer(
        rejected_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_enc["input_ids"][0],
        "chosen_attention_mask": chosen_enc["attention_mask"][0],
        "rejected_input_ids": rejected_enc["input_ids"][0],
        "rejected_attention_mask": rejected_enc["attention_mask"][0],
    }

# ──────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────
def build_ultrafeedback_dataloader(path, tokenizer, batch_size=16):
    ds = load_from_disk(str(path))
    # ds = ds.select(range(200))
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    ds = ds.map(functools.partial(tokenize_ultrafeedback_BERT, tokenizer=tokenizer), batched=False, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)

def load_validation_dataloader(path, tokenizer, batch_size=16):
    ds = load_from_disk(str(path))
    ds = ds.select(range(200))
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    ds = ds.map(functools.partial(tokenize_ultrafeedback_BERT, tokenizer=tokenizer), batched=False, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["chosen_input_ids", "chosen_attention_mask", "rejected_input_ids", "rejected_attention_mask"])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

# ──────────────────────────────────────────────────────────────
# Loss + Training
# ──────────────────────────────────────────────────────────────
def bradley_terry_loss(r_win: torch.Tensor, r_lose: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(r_win - r_lose).mean()

@torch.no_grad()
def evaluate_reward_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        chosen = model(input_ids=batch["chosen_input_ids"].to(device), attention_mask=batch["chosen_attention_mask"].to(device)).logits.squeeze(-1)
        rejected = model(input_ids=batch["rejected_input_ids"].to(device), attention_mask=batch["rejected_attention_mask"].to(device)).logits.squeeze(-1)
        correct += (chosen > rejected).sum().item()
        total += chosen.size(0)
    return correct / total

import pandas as pd

def train_reward_model(model, train_loader, val_loader, optimizer, device, epochs=3):
    model.train()
    logs = []

    for epoch in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for step, batch in enumerate(train_bar):
            chosen = model(input_ids=batch["chosen_input_ids"].to(device), attention_mask=batch["chosen_attention_mask"].to(device)).logits.squeeze(-1)
            rejected = model(input_ids=batch["rejected_input_ids"].to(device), attention_mask=batch["rejected_attention_mask"].to(device)).logits.squeeze(-1)

            loss = bradley_terry_loss(chosen, rejected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (chosen > rejected).sum().item()
            seen += chosen.size(0)
            train_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / (step + 1), acc=correct / seen)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / seen

        # Validation
        model.eval()
        val_loss, val_correct, val_seen = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                chosen = model(input_ids=batch["chosen_input_ids"].to(device), attention_mask=batch["chosen_attention_mask"].to(device)).logits.squeeze(-1)
                rejected = model(input_ids=batch["rejected_input_ids"].to(device), attention_mask=batch["rejected_attention_mask"].to(device)).logits.squeeze(-1)

                loss = bradley_terry_loss(chosen, rejected)
                val_loss += loss.item()
                val_correct += (chosen > rejected).sum().item()
                val_seen += chosen.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_seen

        logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch+1} Summary: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        model.train()

    pd.DataFrame(logs).to_csv("./checkpoints/reward/sBERT_training_log.csv", index=False)
    print("Training log saved to training_log.csv")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"])
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
    model.resize_token_embeddings(len(tokenizer))

    train_loader = build_ultrafeedback_dataloader("./data/ultrafeedback_binarized/train_prefs", tokenizer, batch_size=16)
    val_loader = build_ultrafeedback_dataloader("./data/ultrafeedback_binarized/test_prefs", tokenizer, batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_reward_model(model, train_loader, val_loader, optimizer, device)


    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Model + tokenizer saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
