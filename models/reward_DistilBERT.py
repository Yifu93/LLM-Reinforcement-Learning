import functools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    default_data_collator,
)

# ──────────────────────────────────────────────────────────────
# 1.  Siamese reward model
# ──────────────────────────────────────────────────────────────


class SiameseRewardModel(nn.Module):
    """DistilBERT encoder → MLP head that outputs a single reward score."""

    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden = self.encoder.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # scalar reward
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # [B]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])  # CLS token
        return self.reward_head(cls).squeeze(-1)  # [B]

    def get_rewards(self, win_inputs: dict, lose_inputs: dict):
        """Returns reward_win, reward_lose (each shape [B])."""
        r_win = self(**win_inputs)
        r_lose = self(**lose_inputs)
        return r_win, r_lose


# ──────────────────────────────────────────────────────────────
# 2.  Tokenisation util (Qwen‑style prompt formatting)
# ──────────────────────────────────────────────────────────────


def tokenize_ultrafeedback_BERT(example, *, tokenizer, max_length: int = 512):
    """Convert one UF entry → {chosen_*, rejected_*}"""
    prompt = example["prompt"]
    chosen = example["chosen"][1]["content"]
    rejected = example["rejected"][1]["content"]

    # Qwen chat formatting – kept as literal text tokens.
    base = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
    )

    chosen_text = base + f"<|im_start|>assistant\n{chosen}<|im_end|>"
    rejected_text = base + f"<|im_start|>assistant\n{rejected}<|im_end|>"

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
# 3.  Dataloader builder
# ──────────────────────────────────────────────────────────────


def build_ultrafeedback_dataloader(
    path: str | Path,
    tokenizer: DistilBertTokenizer,
    batch_size: int = 8,
):
    ds = load_from_disk(str(path))
    ds = ds.select(range(100))
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])

    token_fn = functools.partial(tokenize_ultrafeedback_BERT, tokenizer=tokenizer)
    ds = ds.map(token_fn, batched=False, remove_columns=ds.column_names)

    ds.set_format(
        type="torch",
        columns=[
            "chosen_input_ids",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_attention_mask",
        ],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)


# ──────────────────────────────────────────────────────────────
# 4.  Loss + training loop
# ──────────────────────────────────────────────────────────────


def bradley_terry_loss(r_win: torch.Tensor, r_lose: torch.Tensor) -> torch.Tensor:
    """
    L = - log \sigma (r_win - r_lose).
    """
    return -F.logsigmoid(r_win - r_lose).mean()


def train_reward_model(
    model: SiameseRewardModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 3,
):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(bar):
            win_inputs = {
                "input_ids": batch["chosen_input_ids"].to(device),
                "attention_mask": batch["chosen_attention_mask"].to(device),
            }
            lose_inputs = {
                "input_ids": batch["rejected_input_ids"].to(device),
                "attention_mask": batch["rejected_attention_mask"].to(device),
            }

            r_win, r_lose = model.get_rewards(win_inputs, lose_inputs)
            loss = bradley_terry_loss(r_win, r_lose)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # stats
            batch_loss = loss.item()
            total_loss += batch_loss
            correct += (r_win > r_lose).sum().item()
            seen += r_win.size(0)
            bar.set_postfix(Loss=f"{batch_loss:.4f}", AvgLoss=f"{total_loss / (step + 1):.4f}", Acc=f"{correct / seen:.4f}")

        print(
            f"\nEpoch {epoch + 1} finished — AvgLoss: {total_loss / len(dataloader):.4f} | Acc: {correct / seen:.4f}\n"
        )


# ──────────────────────────────────────────────────────────────
# 5.  Main entry point
# ──────────────────────────────────────────────────────────────


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1️⃣  Tokeniser + special Qwen markers
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"])

    # 2️⃣  Model (resize after adding tokens)
    model = SiameseRewardModel().to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))

    # 3️⃣  DataLoader
    dataloader = build_ultrafeedback_dataloader(
        path="./data/ultrafeedback_binarized/train_prefs",
        tokenizer=tokenizer,
        batch_size=8,
    )

    # 4️⃣  Optimiser + training
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_reward_model(model, dataloader, optimizer, device, epochs=3)

    # 5️⃣  Save
    torch.save(model.state_dict(), "reward_model_trained.pth")
    print("Saved model to reward_model_trained.pth")


if __name__ == "__main__":
    main()
