"""
evaluation.py

Run inference with vLLM on a base model and a fine-tuned model and score them.

Tasks
------
• math  -> Countdown (exact-match using countdown.py)
• text  -> SmolTalk (LLM-judge reward)

CLI
---
python -m scripts.evaluation --task math --model-path checkpoints/SFT_WarmStart --out outputs/sft_math_score.json
python -m scripts.evaluation --task text --model-path checkpoints/merged_SmolTak --out outputs/sft_text_score.json
"""

from __future__ import annotations

import argparse, json, sys, pathlib, random
from pathlib import Path
from typing import List, Dict
import gc
import torch

# ──────────────────────────────────────────────────────────────
# 3rd-party deps
from datasets import load_from_disk
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except ImportError as e:  # pragma: no cover
    raise SystemExit("vLLM is required – pip install vllm") from e

from openai import OpenAI

# ──────────────────────────────────────────────────────────────
# countdown scorer  (functions/countdown.py)
from functions.countdown import compute_score


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser("Evaluate a fine-tuned checkpoint against a base one")
    p.add_argument("--task", choices=["math", "text"], required=True)
    p.add_argument("--model-path", required=True, help="fine-tuned checkpoint dir")
    p.add_argument("--base-path", default="checkpoints/initial", help="baseline checkpoint")
    p.add_argument("--out", type=Path, required=True, help="JSON file for the fine-tuned run")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════
# Dataset paths
# ══════════════════════════════════════════════════════════════
EVAL_DATASETS = {
    "math": "data/Countdown-Tasks-3to4/train",
    "text": "data/smoltalk/test",
}

# ══════════════════════════════════════════════════════════════
# Prompt building
# ══════════════════════════════════════════════════════════════
def extract_math_prompts(ds):
    """
    Returns
    -------
    prompts : list[str]
    meta    : list[dict]   ← needed later for scoring
    """
    prompts, meta = [], []
    for ex in ds:
        nums    = ex["nums"]
        target  = ex["target"]
        prompt  = (f"A conversation between User and Assistant. "
                   f"The user asks a question, and the Assistant solves it. "
                   f"The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                   f"User: Using the numbers {nums}, create an equation that equals {target}. "
                   f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                   f"Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
                   f"for example <answer> (1 + 2) / 3 </answer>. Assistant: Let me solve this step by step.")
        prompts.append(prompt)
        meta.append({"numbers": nums, "target": target})
    return prompts, meta


def extract_text_prompts(ds):
    # SmolTalk stores a single user turn in messages[0]
    prompts = [ex["messages"][0]["content"] if ex["messages"] else "" for ex in ds]
    return prompts, [{} for _ in prompts]          # dummy meta


def build_prompt(tok: AutoTokenizer, user_msg: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",    "content": user_msg},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ══════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════
def normalize_dtype(name: str):
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[name.lower()]


def chunk(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def generate_batch(
    llm: LLM,
    tok: AutoTokenizer,
    prompts: List[str],
    params: SamplingParams,
    batch_size: int,
    task,
) -> List[str]:
    replies: List[str] = []
    for group in chunk(prompts, batch_size):
        texts = [build_prompt(tok, p) for p in group]
        outs  = llm.generate(texts, params)
        for out in outs:
            answer = out.outputs[0].text.split("<|im_end|>")[0].strip()
            if task == "math":
                replies.append(f"<|im_start|>assistant\n{answer}")
            else:
                replies.append(answer)  # ultrafeedback replies already formatted
    return replies

# ══════════════════════════════════════════════════════════════
# Scorers
# ══════════════════════════════════════════════════════════════
def score_countdown_math(reply: str, target: int, nums: list[int]) -> float:
    gt = {"target": target, "numbers": nums}
    return compute_score(reply, gt)


def score_llama_judge(client: OpenAI, user: str, assistant: str) -> float:
    msgs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-reward",
        messages=msgs,
    )
    reward_text = completion.choices[0].message.content   # e.g. "reward:-18.73"
    return float(reward_text.split(":")[-1].strip())


# ══════════════════════════════════════════════════════════════
# JSON helpers
# ══════════════════════════════════════════════════════════════
def save_text_json(path: Path,
                   prompts: List[str],
                   replies: List[str],
                   scores: List[float]):
    """
    SmolTalk format:
      [{id, messages:[{role,user} …]}]
    """
    chats = []
    for i, (u, a, s) in enumerate(zip(prompts, replies, scores)):
        chats.append({
            "id": i,
            "messages": [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ],
            "score": s,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chats, ensure_ascii=False, indent=2))
    print(f">>>> wrote {path} ({len(chats)} items)")


def save_math_json(path: Path,
                   replies: List[str],
                   meta: List[dict],
                   scores: List[float]):
    """
    Each record: {raw_reply, target, numbers, score}
    """
    records = []
    for i, (r, m, sc) in enumerate(zip(replies, meta, scores)):
        records.append({
            "id": i,
            "raw_reply": r,
            "target": m["target"],
            "numbers": m["numbers"],
            "score": sc,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f">>>> wrote {path} ({len(records)} items)")


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # Dataset & prompts ---------------------------------------------------------
    ds_path = EVAL_DATASETS[args.task]
    ds      = load_from_disk(ds_path)
    ds      = ds.select(random.sample(range(len(ds)), min(200, len(ds))))  # sample 200 items

    if args.task == "math":
        prompts, meta = extract_math_prompts(ds)
    else:
        prompts, meta = extract_text_prompts(ds)

    # Tokenizer -----------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # vLLM engines --------------------------------------------------------------
    samp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["<|im_end|>"],
    )
    dtype = normalize_dtype(args.dtype)

    print("Launching vLLM back-ends …")

    # 1. Load base model, generate responses, then unload
    print("Loading base model …")
    llm_base = LLM(model="./checkpoints/initial", trust_remote_code=True, dtype=dtype)

    print("Generating responses with base model …")
    base_replies = generate_batch(llm_base, tok, prompts, samp, args.batch, args.task)

    # Unload base model to free memory
    del llm_base
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Load fine-tuned model, generate responses, then unload
    if args.model_path == "./checkpoints/initial":
        print(">>>! Fine-tuned path is same as base — reusing base replies.")
        ft_replies = base_replies
    else:
        print(">>> Loading fine-tuned model …")
        llm_ft = LLM(model=args.model_path, trust_remote_code=True, dtype=dtype)

        print(">>> Generating responses with fine-tuned model …")
        ft_replies = generate_batch(llm_ft, tok, prompts, samp, args.batch, args.task)

        # Unload fine-tuned model (optional, if not needed further)
        del llm_ft
        torch.cuda.empty_cache()
        gc.collect()

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────────
    if args.task == "math":
        base_scores = [
            score_countdown_math(r, m["target"], m["numbers"])
            for r, m in zip(base_replies, meta)
        ]
        ft_scores = [
            score_countdown_math(r, m["target"], m["numbers"])
            for r, m in zip(ft_replies, meta)
        ]
        # aggregate
        print(f"Base mean accuracy : {sum(base_scores)/len(base_scores):.3f}")
        print(f"Fine-tuned mean    : {sum(ft_scores)/len(ft_scores):.3f}")

        # save
        save_math_json(args.out.with_name("base_math.json"), base_replies, meta, base_scores)
        save_math_json(args.out,                             ft_replies,   meta, ft_scores)

    else:  # text
        # one OpenAI client for all calls
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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",
        )

        def build_msgs(prompt, reply):
            return [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": reply}]

        base_scores = [
            score_llama_judge(client, prompt, reply)
            for prompt, reply in zip(prompts, base_replies)
        ]
        ft_scores = [
            score_llama_judge(client, prompt, reply)
            for prompt, reply in zip(prompts, ft_replies)
        ]

        win_rate = sum(1 for i, b in zip(ft_scores, base_scores) if i > b) / len(base_scores)
        print(f">>>> win-rate = {win_rate:.3f}")

        # save
        save_text_json(args.out.with_name("base_text.json"), prompts, base_replies, base_scores)
        save_text_json(args.out,                             prompts, ft_replies,   ft_scores)


if __name__ == "__main__":
    main()
