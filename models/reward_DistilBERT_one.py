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

SAVE_PATH = "./checkpoints/reward/BERT_one"
MODEL_NAME = "distilbert-base-uncased"

# ──────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────
def tokenize_ultrafeedback_BERT(example, *, tokenizer, max_length: int = 512):
    prompt = example["prompt"]
    chosen = example["chosen"][1]["content"]
    rejected = example["rejected"][1]["content"]

    score_chosen = float(example["score_chosen"])
    score_rejected = float(example["score_rejected"])


    base = "system\nYou are a helpful assistant.\nuser\n" + prompt.strip() + "\nassistant\n"

    chosen_text = base + chosen.strip()
    rejected_text = base + rejected.strip()

    chosen_enc = tokenizer(
        chosen_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    rejected_enc = tokenizer(
        rejected_text, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )

    return {
        "chosen_input_ids": chosen_enc["input_ids"][0],
        "chosen_attention_mask": chosen_enc["attention_mask"][0],
        "chosen_score": torch.tensor(score_chosen, dtype=torch.float),
        "rejected_input_ids": rejected_enc["input_ids"][0],
        "rejected_attention_mask": rejected_enc["attention_mask"][0],
        "rejected_score": torch.tensor(score_rejected, dtype=torch.float),
    }

# ──────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────
def build_ultrafeedback_dataloader(path, tokenizer, batch_size=16):
    ds = load_from_disk(str(path))
    # ds = ds.select(range(200))
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    ds = ds.map(functools.partial(tokenize_ultrafeedback_BERT, tokenizer=tokenizer), batched=False, remove_columns=ds.column_names)
    ds.set_format(
        type="torch",
        columns=[
            "chosen_input_ids", "chosen_attention_mask", "chosen_score",
            "rejected_input_ids", "rejected_attention_mask", "rejected_score"],)

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)

def load_validation_dataloader(path, tokenizer, batch_size=16):
    ds = load_from_disk(str(path))
    ds = ds.select(range(200))
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    ds = ds.map(functools.partial(tokenize_ultrafeedback_BERT, tokenizer=tokenizer), batched=False, remove_columns=ds.column_names)
    ds.set_format(
        type="torch",
        columns=[
            "chosen_input_ids", "chosen_attention_mask", "chosen_score",
            "rejected_input_ids", "rejected_attention_mask", "rejected_score"],)

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)

# ──────────────────────────────────────────────────────────────
# Loss + Training
# ──────────────────────────────────────────────────────────────
# def bradley_terry_loss(r_win: torch.Tensor, r_lose: torch.Tensor) -> torch.Tensor:
#     return -F.logsigmoid(r_win - r_lose).mean()

def mse_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)

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


def train_reward_model(model, train_loader, val_loader, optimizer, device, epochs=10):
    logs = []

    for epoch in range(epochs):
        model.train()
        total_loss, total_examples = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in train_bar:
            inputs_chosen = { "input_ids": batch["chosen_input_ids"].to(device), "attention_mask": batch["chosen_attention_mask"].to(device) }
            inputs_rejected = { "input_ids": batch["rejected_input_ids"].to(device), "attention_mask": batch["rejected_attention_mask"].to(device) }

            scores_chosen = batch["chosen_score"].to(device)
            scores_rejected = batch["rejected_score"].to(device)

            preds_chosen = model(**inputs_chosen).logits.squeeze(-1)
            preds_rejected = model(**inputs_rejected).logits.squeeze(-1)

            loss = mse_regression_loss(preds_chosen, scores_chosen) + mse_regression_loss(preds_rejected, scores_rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_examples += scores_chosen.size(0)
            train_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / total_examples)

        avg_train_loss = total_loss / total_examples

        # Eval
        model.eval()
        val_loss, val_examples = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs_chosen = { "input_ids": batch["chosen_input_ids"].to(device), "attention_mask": batch["chosen_attention_mask"].to(device) }
                inputs_rejected = { "input_ids": batch["rejected_input_ids"].to(device), "attention_mask": batch["rejected_attention_mask"].to(device) }

                scores_chosen = batch["chosen_score"].to(device)
                scores_rejected = batch["rejected_score"].to(device)

                preds_chosen = model(**inputs_chosen).logits.squeeze(-1)
                preds_rejected = model(**inputs_rejected).logits.squeeze(-1)

                loss = mse_regression_loss(preds_chosen, scores_chosen) + mse_regression_loss(preds_rejected, scores_rejected)
                val_loss += loss.item()
                val_examples += scores_chosen.size(0)

        avg_val_loss = val_loss / val_examples

        logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        })

        print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    pd.DataFrame(logs).to_csv("BERT_regression_training_log.csv", index=False)
    print("Saved log to BERT_regression_training_log.csv")


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
    val_loader = load_validation_dataloader("./data/ultrafeedback_binarized/test_prefs", tokenizer, batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_reward_model(model, train_loader, val_loader, optimizer, device)


    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Model + tokenizer saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
