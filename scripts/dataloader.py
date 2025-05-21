# For each dataset, this script will create a dataloader for the dataset
# -- Preference Dataset --
# (1) SmolTalk (Dataset for SFT): https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
# (2) UltraFeedback (Dataset for DPO and RLOO): https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
# -- Verifier-Based Dataset --
# (3) WarmStart (Dataset for SFT): https://huggingface.co/datasets/Asap7772/cog_behav_all_strategies
# (4) PromptsDataset from TinyZero (RLOO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
# (5) On-Policy Preference Dataset (the same as PromptsDataset, DPO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4


from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from models.qwen_model import load_tokenizer
from transformers import default_data_collator

# Constants
SAVE_DIR = "./qwen2_model"
BATCH_SIZE = 8
MAX_LENGTH = 1024  # Max length for the model

# Load tokenizer
tokenizer = load_tokenizer(SAVE_DIR)

# ──────────────────────────────────────────────────────────────
# SmolTalk Dataset -- for SFT
# (1) Extract only the first user/assistant turn
def select_from_smoltalk(example):
    messages = example.get("messages", [])
    if len(messages) >= 1 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
        # print("Selecting from SmolTalk:", messages[1]["content"])
        return {
            "prompt": messages[0]["content"],
            "response": messages[1]["content"],
        }
    else:
        return {
            "prompt": None,
            "response": None,
            }

# (2) Tokenize with chat template
def tokenize_SmolTalk_sft(example):
    # Prepare structured messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]

    # Tokenize prompt-only (system + user) without padding
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True  # ends at <|im_start|>assistant\n
    )
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)

    # Tokenize full prompt + response
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_tokens = tokenizer(full_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)

    input_ids = full_tokens["input_ids"][0]
    attention_mask = full_tokens["attention_mask"][0]
    position_ids = torch.cumsum(attention_mask, dim=0) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    labels = input_ids.clone()

    # Mask out the prompt portion
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

    ds = ds.map(
        tokenize_SmolTalk_sft,
        batched=False,
        remove_columns=ds.column_names,
    )

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "position_ids", "labels"])
    return ds


# (3) Build the smoltalk DataLoader
def get_smoltalk_dataloader(path="./data/smoltalk/train"):
    ds = load_from_disk(path)
    # print("Before filtering:", ds)

    # Extract prompt/response
    ds = ds.map(select_from_smoltalk)
    # print("Selected:", ds[0]["prompt"])

    # Drop blank ones
    ds = ds.filter(lambda x: x["prompt"] and x["response"])
    # print("After filtering:", ds[0]["prompt"])

    # Tokenize
    ds = ds.map(
        tokenize_SmolTalk_sft,
        batched=False,
        remove_columns=ds.column_names,
    )

    # Convert to PyTorch tensors
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

def get_warmstart_dataset(path="./data/warmstart/train"):
    ds = load_from_disk(path)
    ds = ds.filter(lambda x: x["query"] and x["completion"])

    ds = ds.map(
        tokenize_WarmStart_sft,
        batched=False,
        remove_columns=ds.column_names,
    )

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "position_ids", "labels"])
    return ds

# ──────────────────────────────────────────────────────────────
# UltraFeedback Dataset -- for DPO

# (1) Tokenize for DPO
def tokenize_ultrafeedback_dpo(example):
    prompt = example["prompt"]
    chosen = example["chosen"][1]['content']
    rejected = example["rejected"][1]['content']
    # print("Tokenizing ultrafeedback example:", chosen)
    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    # Tokenize prompt only (to get prompt length)
    prompt_text = tokenizer.apply_chat_template(
        base_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    )
    prompt_len = prompt_tokens["input_ids"].shape[1]

    # Chosen
    chosen_text = tokenizer.apply_chat_template(
        base_messages + [{"role": "assistant", "content": chosen}],
        tokenize=False,
        add_generation_prompt=False
    )
    chosen = tokenizer(
        chosen_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    # Rejected
    rejected_text = tokenizer.apply_chat_template(
        base_messages + [{"role": "assistant", "content": rejected}],
        tokenize=False,
        add_generation_prompt=False
    )
    rejected = tokenizer(
        rejected_text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )

    return {
        "chosen_input_ids": chosen["input_ids"][0],
        "chosen_attention_mask": chosen["attention_mask"][0],
        "rejected_input_ids": rejected["input_ids"][0],
        "rejected_attention_mask": rejected["attention_mask"][0],
        "prompt_length": prompt_len
    }

# (2) Build the DataLoader
def get_ultrafeedback_dataloader_dpo(path="./data/ultrafeedback_binarized/train_prefs"):
    ds = load_from_disk(path)
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    ds = ds.map(tokenize_ultrafeedback_dpo, batched=False, remove_columns=ds.column_names)

    ds.set_format(type="torch", columns=[
        "chosen_input_ids", "chosen_attention_mask",
        "rejected_input_ids", "rejected_attention_mask",
        "prompt_length"
    ])
    # During DPO loss computation, the prompt portion (which is identical for chosen and rejected) 
    # is ignored when computing reward differences. 
    # prompt_length helps the trainer know where to start scoring.
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# ──────────────────────────────────────────────────────────────
# WarmStart Dataset -- for SFT
# (1) Tokenize with chat template
def tokenize_WarmStart_sft(example):
    # Prepare structured messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["completion"]},
    ]

    # Tokenize prompt-only (system + user), used to calculate prompt_len
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True  # ends with <|im_start|>assistant\n
    )
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt", add_special_tokens=False)

    # Tokenize full prompt + response
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=False
    )

    input_ids = full_tokens["input_ids"][0]
    attention_mask = full_tokens["attention_mask"][0]

    # Compute position_ids: needed for Qwen2 rotary embeddings
    position_ids = torch.cumsum(attention_mask, dim=0) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)

    # Create labels and mask prompt portion
    labels = input_ids.clone()
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:prompt_len] = -100  # ignore prompt in loss

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "position_ids": position_ids.tolist(),
        "labels": labels.tolist(),
    }

# (2) Build the WarmStart DataLoader
def get_WarmStart_dataloader(path="./data/warmstart/train"):
    ds = load_from_disk(path)

    # Drop blank ones
    ds = ds.filter(lambda x: x["query"] and x["completion"])
    # print("After filtering:", ds[0]["query"])

    # Tokenize
    ds = ds.map(
        tokenize_WarmStart_sft,
        batched=False,
        remove_columns=ds.column_names,
    )

    # Convert to PyTorch tensors
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)




# ──────────────────────────────────────────────────────────────
# TinyZero Dataset -- for DPO




# ──────────────────────────────────────────────────────────────
# TinyZero Dataset -- for RLOO




if __name__ == "__main__":
    # (1) Load smolTalk dataset (for SFT)
    smoltalk_train_path = "./data/smoltalk/train"
    smoltalk_dl = get_smoltalk_dataloader(smoltalk_train_path)
    print(">>>>>>>> Dataloader created for smoltalk training.")

    # (2) Load ultrafeedback dataset (for DPO)
    ultrafeedback_train_path = "./data/ultrafeedback_binarized/train_prefs"
    ultrafeedback_dl = get_ultrafeedback_dataloader_dpo(ultrafeedback_train_path)
    print(">>>>>>>> Dataloader created for ultrafeedback training.")

    # (3) Load ultrafeedback dataset (for RLOO)
    # TODO: to be implemented

    # (4) Load warmstart dataset (for SFT)
    warmstart_train_path = "./data/warmstart/train"
    warmstart_dl = get_WarmStart_dataloader(warmstart_train_path)
    print(">>>>>>>> Dataloader created for warmstart training.")
    # (5) Load countdown dataset (for DPO/RLOO)
    # TODO: to be implemented

    # # -----------------------
    # # Use for testing SFT
    # batch = next(iter(warmstart_dl))
    # print("Batch keys:", batch.keys())

    # # Check batch shapes
    # print("input_ids shape:", batch["input_ids"].shape)
    # print("attention_mask shape:", batch["attention_mask"].shape)
    # print("labels shape:", batch["labels"].shape)

    # sample_input_ids = batch["input_ids"][0]
    # sample_labels = batch["labels"][0]

    # decoded_input = tokenizer.decode(
    #     [id for id in sample_input_ids if id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )
    # decoded_labels = tokenizer.decode(
    #     [id for id in sample_labels if id != -100 and id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )

    # print("\nDecoded input (prompt + response):")
    # print(decoded_input)

    # print("\nDecoded labels (only response):")
    # print(decoded_labels)

    # # -----------------------
    # # Test for DPO
    # batch = next(iter(ultrafeedback_dl))
    # print("Batch keys:", batch.keys())

    # # Check batch shapes
    # print("chosen_input_ids shape:", batch["chosen_input_ids"].shape)
    # print("rejected_input_ids shape:", batch["rejected_input_ids"].shape)
    # print("prompt_length shape:", batch["prompt_length"].shape)

    # # Take first sample in batch
    # chosen_ids = batch["chosen_input_ids"][0]
    # rejected_ids = batch["rejected_input_ids"][0]
    # prompt_len = batch["prompt_length"][0].item()

    # # Decode full sequences
    # decoded_chosen = tokenizer.decode(
    #     [id for id in chosen_ids if id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )
    # decoded_rejected = tokenizer.decode(
    #     [id for id in rejected_ids if id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )

    # # Decode prompt only
    # decoded_prompt = tokenizer.decode(
    #     [id for id in chosen_ids[:prompt_len] if id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )

    # # Decode only the chosen/rejected response
    # decoded_chosen_resp = tokenizer.decode(
    #     [id for id in chosen_ids[prompt_len:] if id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )
    # decoded_rejected_resp = tokenizer.decode(
    #     [id for id in rejected_ids[prompt_len:] if id != tokenizer.pad_token_id],
    #     skip_special_tokens=True
    # )

    # # Display
    # print("\nDecoded prompt:")
    # print(decoded_prompt)

    # print("\nDecoded chosen response:")
    # print(decoded_chosen_resp)

    # print("\nDecoded rejected response:")
    # print(decoded_rejected_resp)