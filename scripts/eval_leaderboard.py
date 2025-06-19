"""
eval_leaderboard.py  ·  generate leaderboard submissions + scores
---------------------------------------------------------------

Usage examples
--------------
python -m scripts.eval_leaderboard      \
        --task math                     \
        --model-path checkpoints/SFT_WarmStart \
        --out outputs/submit_math.json

python -m scripts.eval_leaderboard      \
        --task text                     \
        --model-path checkpoints/merged_SmolTalk \
        --out outputs/submit_text.json
"""

from __future__ import annotations
import argparse, json, gc, re
from pathlib import Path
from typing import List, Dict, Any, Iterable

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from openai import OpenAI        
from functions.countdown import compute_score, extract_solution

# ──────────────────────────────────────────────────────────────

# Location of the raw eval files (plain JSON, **not** dataset-dict)
EVAL_DATASETS = {
    # "math": "data/leader_board/countdown.json",
    # "text": "data/leader_board/ultrafeedback.json",
    "text": "data/leader_board/ultrafeedback_heldout_prompt.json",
    "math": "data/leader_board/countdown_heldout_prompts.json",
}

# ──────────────────────────────────────────────────────────────
# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["math", "text"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--base-path", default="checkpoints/initial")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────
# Helpers
def normalize_dtype(x: str) -> str:
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[x]

def chunk(seq: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

# ──────────────────────────────────────────────────────────────
# Prompt extraction
def load_json(path: str) -> list[dict]:
    """Load either a JSON array or JSON-Lines file transparently."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # JSON-Lines starts with "{" or whitespace then "{"
        if first.strip().startswith("{"):
            return [json.loads(line) for line in f if line.strip()]
        else:                        # normal JSON array
            return json.load(f)


def extract_math_prompts(raw: list[dict]) -> tuple[list[str], list[dict]]:
    prompts, meta = [], []
    for ex in raw:
        nums, tgt = ex["num"], ex["target"]
        prompt = (f"A conversation between User and Assistant. "
                  f"The user asks a question, and the Assistant solves it. "
                  f"The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                  f"User: Using the numbers {nums}, create an equation that equals {tgt}. "
                  f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                  f"Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
                  f"for example <answer> (1 + 2) / 3 </answer>. Assistant: Let me solve this step by step.")
        prompts.append(prompt)
        meta.append({"numbers": nums, "target": tgt})
    return prompts, meta

def extract_text_prompts(raw: list[dict]) -> tuple[list[str], list[dict]]:
    return [ex["prompt"] for ex in raw], [{} for _ in raw]

def build_prompt(tok: AutoTokenizer, user_msg: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_msg},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ──────────────────────────────────────────────────────────────
# Generation
def generate_replies(
    llm: LLM,
    tok: AutoTokenizer,
    user_prompts: list[str],
    params: SamplingParams,
    batch: int,
    task
) -> list[str]:
    replies: list[str] = []
    for grp in chunk(user_prompts, batch):
        full_prompts = [build_prompt(tok, p) for p in grp]
        outs = llm.generate(full_prompts, params)
        for o in outs:
            ans = o.outputs[0].text.split("<|im_end|>")[0].strip()
            if task == "math":
                replies.append(f"<|im_start|>assistant\n{ans}")
            else:
                replies.append(ans)  # ultrafeedback replies already formatted
    return replies

# ──────────────────────────────────────────────────────────────
# Scoring
def score_math(replies: list[str], meta: list[dict]) -> list[float]:
    return [compute_score(r, m) for r, m in zip(replies, meta)]

def score_text_llama(client: OpenAI, prompts: list[str], replies: list[str]) -> list[float]:
    def one(u, a):
        res = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=[{"role":"user","content":u},
                      {"role":"assistant","content":a}],
        )
        txt = res.choices[0].message.content
        return float(txt.split(":")[-1])
    return [one(u, a) for u, a in zip(prompts, replies)]

# ──────────────────────────────────────────────────────────────
# Writers
def write_math(path: Path,
               replies: List[str],
               meta:   List[Dict]):
    """
    Submission format (one JSON object per line):

      {"num":[1,2,3], "target":6, "response":"1+2+3=6"}
      …

    • `replies[i]` is the raw assistant text (already stripped of <|im_end|>)
    • `meta[i]` contains {"numbers":…, "target":…}
    • We ignore scores – leaderboard only needs the equation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for r, m in zip(replies, meta):
            expr = extract_solution(r) or ""   # fallback to empty string
            record = {
                "num":   m["numbers"],
                "target": m["target"],
                "response": expr,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f">>> saved {len(replies)} math solutions --> {path}")


def write_text(path: Path,
               prompts: List[str],
               replies: List[str]):
    """
    Submission format (one JSON object per line):

      {"prompt":"Prompt 1.","response":"My response 1."}
      {"prompt":"Prompt 2.","response":"My response 2."}
      …

    No scores or IDs required by the leaderboard.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for p, r in zip(prompts, replies):
            f.write(json.dumps({"prompt": p, "response": r},
                               ensure_ascii=False) + "\n")

    print(f">>> saved {len(replies)} text responses --> {path}")

# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    raw = load_json(EVAL_DATASETS[args.task])

    prompts, meta = (
        extract_math_prompts(raw) if args.task == "math"
        else extract_text_prompts(raw)
    )

    tok = AutoTokenizer.from_pretrained(args.base_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    samp = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        stop=["<|im_end|>"],
    )
    dtype = normalize_dtype(args.dtype)

    # ---------- Base model
    base_llm = LLM(model=args.base_path, trust_remote_code=True, dtype=dtype)
    base_replies = generate_replies(base_llm, tok, prompts, samp, args.batch, args.task)
    del base_llm; torch.cuda.empty_cache(); gc.collect()

    # ---------- Fine-tuned model
    if args.model_path == args.base_path:
        ft_replies = base_replies
    else:
        ft_llm = LLM(model=args.model_path, trust_remote_code=True, dtype=dtype)
        ft_replies = generate_replies(ft_llm, tok, prompts, samp, args.batch, args.task)
        del ft_llm; torch.cuda.empty_cache(); gc.collect()

    # ---------- Scoring & saving
    if args.task == "math":
        ft_scores  = score_math(ft_replies, meta)
        base_scores = score_math(base_replies, meta)
        print(f"FT math accuracy: {sum(ft_scores)/len(ft_scores):.3f} "
              f"(base {sum(base_scores)/len(base_scores):.3f})")
        write_math(args.out.with_name("base_math.json"), base_replies, meta)
        write_math(args.out, ft_replies, meta)

    else:  # text
        client = OpenAI(  # your NVIDIA llama-judge endpoint
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="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",
        )
        ft_scores   = score_text_llama(client, prompts, ft_replies)
        base_scores = score_text_llama(client, prompts, base_replies)
        win = sum(int(f > b) for f, b in zip(ft_scores, base_scores)) / len(ft_scores)
        print(f"FT win-rate vs base: {win:.3f}")
        write_text(args.out.with_name("base_text.json"), prompts, base_replies)
        write_text(args.out, prompts, ft_replies)

if __name__ == "__main__":
    main()
