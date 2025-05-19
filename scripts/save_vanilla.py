# scripts/save_vanilla.py
"""
Download the base Qwen-2.5-0.5B weights + tokenizer, save them
to ./checkpoints/initial, then reload from disk and run a sanity check.

Usage:
    python -m scripts.save_vanilla --dtype bf16
"""

import argparse
import torch
from pathlib import Path
import logging

from models.qwen_model import (
    download_tokenizer,
    download_model,
    load_model,
    load_tokenizer,
    save_model,
    save_model_and_tokenizer,
)

# ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ──────────────────────────────────────────────────────────────
SAVE_DIR = Path("./checkpoints/initial")
# DEFAULT_PROMPT = "Hello, how are you today?"
DEFAULT_PROMPT = "Using the numbers [95, 36, 32], create an equation that equals 91. " \
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. " \
    "Show your work in <think> </think> tags. " \
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download base Qwen and test it")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()

# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # # 1️⃣ Download model + tokenizer from Hugging Face
    # tokenizer = load_tokenizer()
    # model = load_model(dtype=args.dtype)

    # # 2️⃣ Save to local directory
    # save_model_and_tokenizer(model, tokenizer, SAVE_DIR)

    # 3️⃣ Reload from saved checkpoint
    tokenizer = load_tokenizer(SAVE_DIR)
    model = load_model(checkpoint_path=SAVE_DIR, dtype=args.dtype)

    # 4️⃣ Minimal generation sanity-check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    messages = [
        {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer."},
        {"role": "user", "content": args.prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    batch = tokenizer(prompt_text, return_tensors="pt").to(device)

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.no_grad():
        gen_ids = model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.1,
        )

    prompt_len = batch["input_ids"].shape[1]
    raw_output = tokenizer.decode(gen_ids[0, prompt_len:], skip_special_tokens=False)
    assistant_reply = raw_output.split("<|im_end|>")[0].strip()

    print("\n─── Assistant reply ───\n")
    print(assistant_reply)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
