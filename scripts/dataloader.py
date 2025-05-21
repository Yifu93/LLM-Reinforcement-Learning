# For each dataset, this script will create a dataloader for the dataset
# -- Preference Dataset --
# (1) SmolTalk (Dataset for SFT): https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
# (2) UltraFeedback (Dataset for DPO and RLOO): https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
# -- Verifier-Based Dataset --
# (3) WarmStart (Dataset for SFT): https://huggingface.co/datasets/Asap7772/cog_behav_all_strategies
# (4) PromptsDataset from TinyZero (RLOO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
# (5) On-Policy Preference Dataset (the same as PromptsDataset, DPO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4


from datasets import load_from_disk
from torch.utils.data import DataLoader
from models.qwen_model import load_tokenizer
from transformers import default_data_collator

# Constants
SAVE_DIR = "./checkpoints/initial"
BATCH_SIZE = 8
MAX_LENGTH = 1024  # Max length for the model

# Load tokenizer
tokenizer = load_tokenizer(SAVE_DIR)

# ──────────────────────────────────────────────────────────────
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
def tokenize_sft(example):
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
    labels = input_ids.clone()

    # Mask out the prompt portion
    prompt_len = prompt_tokens["input_ids"].shape[1]
    labels[:prompt_len] = -100

    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    }


# ──────────────────────────────────────────────────────────────
# (3) Build the DataLoader
def get_smoltalk_dataloader(path="./data/smoltalk/train"):
    ds = load_from_disk(path)
    ds = ds.select(range(2))  # Use more data later if needed
    # print("Before filtering:", ds)

    # Extract prompt/response
    ds = ds.map(select_from_smoltalk)
    # print("Selected:", ds[0]["prompt"])

    # Drop blank ones
    ds = ds.filter(lambda x: x["prompt"] and x["response"])
    # print("After filtering:", ds[0]["prompt"])

    # Tokenize
    ds = ds.map(
        tokenize_sft,
        batched=False,
        remove_columns=ds.column_names,
    )

    # Convert to PyTorch tensors
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # print("Tokenized:", ds)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    smoltalk_train_path = "./data/smoltalk/train"
    smoltalk_dl = get_smoltalk_dataloader(smoltalk_train_path)
    print("Dataloader created for smoltalk training.")

    # batch = next(iter(smoltalk_dl))
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
