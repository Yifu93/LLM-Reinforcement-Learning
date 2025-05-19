
"""Script for SFT / DPO / RLOO training.
Fine tune the model on the different dataset and using different training methods
(1) SFT (Supervised Fine-Tuning)
(2) DPO (Direct Preference Optimization)
(3) RLOO (Reinforcement Learning from Offline Optimization)
Usage:
    python train.py --task sft --data ./data/smoltalk --output ./checkpoints/sft
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from models.qwen_model import ModelConfig, load_model, load_tokenizer
from scripts.dataloader import get_dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Qwen fine‑tune script")
    p.add_argument("--task", choices=["sft", "dpo", "rloo"], required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--output", type=str, default="./checkpoints")
    p.add_argument("--lora_r", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = ModelConfig(lora_r=args.lora_r)

    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg)

    dataloader = get_dataloader(args.data, tokenizer, args.task, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    model.train()

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}

            if args.task == "sft":
                loss = model(**batch).loss
            else:
                # Placeholder for DPO / RLOO losses
                raise NotImplementedError("DPO/RLOO loss not yet implemented")

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        # Save checkpoint each epoch
        ckpt_dir = Path(args.output) / f"{args.task}_epoch{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)


if __name__ == "__main__":
    main()

