from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from models.qwen_model import load_tokenizer
from transformers import default_data_collator

# Constants
SAVE_DIR = "./qwen2_model"
BATCH_SIZE = 4
MAX_LENGTH = 1024 # Maximum sequence length for tokenization

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

def tokenize_SmolTalk_sft_batch(examples):
    prompts = examples["prompt"]
    responses = examples["response"]

    batch_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p},
            {"role": "assistant", "content": r},
        ]
        for p, r in zip(prompts, responses)
    ]

    prompt_texts = [
        tokenizer.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    full_texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in batch_messages
    ]

    prompt_tokenized = tokenizer(prompt_texts, add_special_tokens=False)
    full_tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    input_ids_list, attention_mask_list, position_ids_list, labels_list = [], [], [], []

    for input_ids, attention_mask, prompt_ids in zip(full_tokenized["input_ids"], full_tokenized["attention_mask"], prompt_tokenized["input_ids"]):
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        position_ids = torch.cumsum(attention_mask, dim=0) - 1
        position_ids[attention_mask == 0] = 0

        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100     # Ignore the prompt part in labels
        labels[attention_mask == 0] = -100  # Ensure padding tokens are ignored

        # # Double-check everything:
        # print(f"----------------------------")
        # print('input_ids:', input_ids)
        # print('attention_mask:', attention_mask)
        # print('position_ids:', position_ids)
        # print('labels:', labels)
        # print(f"----------------------------")

        input_ids_list.append(input_ids.tolist())
        attention_mask_list.append(attention_mask.tolist())
        position_ids_list.append(position_ids.tolist())
        labels_list.append(labels.tolist())

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "position_ids": position_ids_list,
        "labels": labels_list,
    }

def get_smoltalk_dataset(path="./data/smoltalk/train"):
    ds = load_from_disk(path)
    ds = ds.map(select_from_smoltalk)
    ds = ds.filter(lambda x: x["prompt"] and x["response"])
    ds = ds.map(tokenize_SmolTalk_sft_batch, batched=True, batch_size=32, remove_columns=list(ds.features))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "position_ids", "labels"])
    return ds

def get_smoltalk_dataloader(path="./data/smoltalk/train"):
    ds = get_smoltalk_dataset(path)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# ──────────────────────────────────────────────
# WarmStart Dataset (SFT)
def tokenize_WarmStart_sft_batch(examples):
    queries = examples["query"]
    completions = examples["completion"]

    batch_messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": c},
        ]
        for q, c in zip(queries, completions)
    ]

    prompt_texts = [
        tokenizer.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    full_texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in batch_messages
    ]

    prompt_tokenized = tokenizer(prompt_texts, add_special_tokens=False)
    full_tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False,
    )

    input_ids_list, attention_mask_list, position_ids_list, labels_list = [], [], [], []

    for input_ids, attention_mask, prompt_ids in zip(full_tokenized["input_ids"], full_tokenized["attention_mask"], prompt_tokenized["input_ids"]):
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        position_ids = torch.cumsum(attention_mask, dim=0) - 1
        position_ids[attention_mask == 0] = 0

        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100     # Ignore the prompt part in labels
        labels[attention_mask == 0] = -100  # Ensure padding tokens are ignored

        input_ids_list.append(input_ids.tolist())
        attention_mask_list.append(attention_mask.tolist())
        position_ids_list.append(position_ids.tolist())
        labels_list.append(labels.tolist())

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "position_ids": position_ids_list,
        "labels": labels_list,
    }

def get_warmstart_dataset(path="./data/warmstart/train", debug=False):
    ds = load_from_disk(path)
    ds = ds.filter(lambda x: x.get("query") and x.get("completion"))
    ds = ds.map(tokenize_WarmStart_sft_batch, batched=True, batch_size=32, remove_columns=list(ds.features))
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
def select_from_ultrafeedback(example):
    chosen = example.get("chosen")
    rejected = example.get("rejected")

    return {
        "chosen": chosen,
        "rejected": rejected
    }

def get_ultrafeedback_dataset(path="./data/ultrafeedback_binarized/train_prefs"):
    ds = load_from_disk(path)
    # ds = ds.select(range(2))  # For debugging, limit to 2 samples
    ds = ds.map(select_from_ultrafeedback)
    ds = ds.filter(lambda x: x["chosen"] and x["rejected"]) # filter out invalid entries
    return ds

def get_ultrafeedback_dataloader_dpo(path="./data/ultrafeedback_binarized/train_prefs"):
    ds = get_ultrafeedback_dataset(path)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# ──────────────────────────────────────────────
# UltraFeedback Dataset (RLOO)
# Placeholder

