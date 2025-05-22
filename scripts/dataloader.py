from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from models.qwen_model import load_tokenizer
from transformers import default_data_collator

# Constants
SAVE_DIR = "./qwen2_model"
BATCH_SIZE = 8
MAX_LENGTH = 1024  # For causal LM context window

# Load tokenizer globally (used in all tokenizers)
tokenizer = load_tokenizer(SAVE_DIR)
tokenizer.pad_token = tokenizer.eos_token


# ──────────────────────────────────────────────
# SmolTalk Dataset (SFT)
def select_from_smoltalk(example):
    messages = example.get("messages", [])
    if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        return {
            "prompt": messages[0]["content"],
            "response": messages[1]["content"],
        }
    return {"prompt": None, "response": None}


def tokenize_SmolTalk_sft(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    prompt_only = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_tokens = tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=MAX_LENGTH, add_special_tokens=False)

    input_ids = full_tokens["input_ids"][0]
    attention_mask = full_tokens["attention_mask"][0]
    position_ids = torch.cumsum(attention_mask, dim=0) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    labels = input_ids.clone()
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "position_ids": position_ids.tolist(),
        "labels": labels.tolist(),
    }


def get_smoltalk_dataset(path="./data/smoltalk/train"):
    ds = load_from_disk(path)
    ds = ds.map(select_from_smoltalk)
    ds = ds.filter(lambda x: x["prompt"] and x["response"])
    ds = ds.map(tokenize_SmolTalk_sft, batched=False, remove_columns=list(ds.features))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "position_ids", "labels"])
    return ds


def get_smoltalk_dataloader(path="./data/smoltalk/train"):
    ds = get_smoltalk_dataset(path)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)


# ──────────────────────────────────────────────
# WarmStart Dataset (SFT)

def tokenize_WarmStart_sft(example):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    prompt_only = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_tokens = tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True,
                            max_length=MAX_LENGTH, add_special_tokens=False)

    input_ids = full_tokens["input_ids"][0]
    attention_mask = full_tokens["attention_mask"][0]
    position_ids = torch.cumsum(attention_mask, dim=0) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    labels = input_ids.clone()
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "position_ids": position_ids.tolist(),
        "labels": labels.tolist(),
    }


def get_warmstart_dataset(path="./data/warmstart/train", debug=False):
    ds = load_from_disk(path)
    ds = ds.filter(lambda x: x.get("query") and x.get("completion"))
    ds = ds.map(tokenize_WarmStart_sft, batched=False, remove_columns=list(ds.features))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "position_ids", "labels"])

    if debug:
        sample = ds[0]
        print("\n[DEBUG] Decoded input:\n", tokenizer.decode([i for i in sample["input_ids"] if i != tokenizer.pad_token_id], skip_special_tokens=True))
        print("[DEBUG] Decoded label (response only):\n", tokenizer.decode([i for i in sample["labels"] if i != -100 and i != tokenizer.pad_token_id], skip_special_tokens=True))
    return ds


def get_warmstart_dataloader(path="./data/warmstart/train"):
    ds = get_warmstart_dataset(path)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)


# ──────────────────────────────────────────────
# UltraFeedback Dataset (DPO)

def tokenize_ultrafeedback_dpo(example):
    prompt = example["prompt"]
    chosen = example["chosen"][1]["content"]
    rejected = example["rejected"][1]["content"]

    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(base_messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = prompt_tokens["input_ids"].shape[1]

    chosen_text = tokenizer.apply_chat_template(base_messages + [{"role": "assistant", "content": chosen}], tokenize=False)
    chosen = tokenizer(chosen_text, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=MAX_LENGTH, add_special_tokens=False)

    rejected_text = tokenizer.apply_chat_template(base_messages + [{"role": "assistant", "content": rejected}], tokenize=False)
    rejected = tokenizer(rejected_text, return_tensors="pt", padding="max_length", truncation=True,
                         max_length=MAX_LENGTH, add_special_tokens=False)

    return {
        "chosen_input_ids": chosen["input_ids"][0],
        "chosen_attention_mask": chosen["attention_mask"][0],
        "rejected_input_ids": rejected["input_ids"][0],
        "rejected_attention_mask": rejected["attention_mask"][0],
        "prompt_length": prompt_len
    }


def get_ultrafeedback_dataloader_dpo(path="./data/ultrafeedback_binarized/train_prefs"):
    ds = load_from_disk(path)
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    ds = ds.map(tokenize_ultrafeedback_dpo, batched=False, remove_columns=list(ds.features))
    ds.set_format(type="torch", columns=[
        "chosen_input_ids", "chosen_attention_mask",
        "rejected_input_ids", "rejected_attention_mask",
        "prompt_length"
    ])
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
