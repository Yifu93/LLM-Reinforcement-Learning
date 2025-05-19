# scripts/generate_test_vllm.py
"""
Run a generation test using vLLM, loading model from ./checkpoints/initial.

Usage:
    python scripts/generate_test_vllm.py --prompt "What is 44 + 35 + 19?" --dtype bf16
"""

import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

SAVE_DIR = "./checkpoints/initial"
DEFAULT_PROMPT = "Using the numbers [95, 36, 32], create an equation that equals 91. " \
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. " \
    "Show your work in <think> </think> tags. " \
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."

def parse_args():
    parser = argparse.ArgumentParser("Test vLLM generation with a local checkpoint")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()

def normalize_dtype(dtype: str) -> str:
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }[dtype]


def main():
    args = parse_args()

    # Load tokenizer for chat formatting
    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Build prompt using Qwen chat format
    messages = [
        {"role": "system", "content": "A conversation between User and Assistant. "
         "The Assistant first thinks about the reasoning process in the mind, then answers."},
        {"role": "user", "content": args.prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # vLLM generation setup
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["<|im_end|>"],  # optional stop string if <|im_end|> isn't respected automatically
    )

    llm = LLM(model=SAVE_DIR,
        trust_remote_code=True,
        dtype=normalize_dtype(args.dtype))

    # Run generation
    outputs = llm.generate([prompt], sampling_params)

    # Decode and print response
    text = outputs[0].outputs[0].text
    reply = text.split("<|im_end|>")[0].strip()

    print("\n─── Assistant reply (vLLM) ───\n")
    print(reply)


if __name__ == "__main__":
    main()
