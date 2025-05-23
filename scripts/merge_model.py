# This script merges a LoRA model into its base model and saves the merged model.
"""
python -m scripts.merge_model \
  --lora_dir checkpoints/SFT_SmolTak \
  --save_path checkpoints/merged_SmolTak
"""

import argparse
from peft import PeftModel, PeftConfig
from models.qwen_model import load_model, save_model

def merge_lora_model(lora_dir: str, save_path: str):
    """
    Merge a LoRA adapter into the base model and save the merged model.
    """
    print(f"🔍 Loading PEFT config from {lora_dir}")
    peft_config = PeftConfig.from_pretrained(lora_dir)

    base_model_name = peft_config.base_model_name_or_path
    print(f"📦 Base model: {base_model_name}")

    base_model = load_model("./checkpoints/initial", dtype="bf16")

    print(f"🔧 Applying LoRA from {lora_dir}")
    lora_model = PeftModel.from_pretrained(base_model, lora_dir)

    print("🧩 Merging LoRA weights into base model …")
    merged_model = lora_model.merge_and_unload()

    print(f"💾 Saving merged model to {save_path}")
    save_model(merged_model, save_path)

    print("✅ Merge complete.")


def parse_args():
    parser = argparse.ArgumentParser("Merge a LoRA model into its base model")
    parser.add_argument("--lora_dir", required=True, help="Directory with the LoRA adapter")
    parser.add_argument("--save_path", required=True, help="Path to save the merged model")
    return parser.parse_args()


def main():
    args = parse_args()
    merge_lora_model(args.lora_dir, args.save_path)


if __name__ == "__main__":
    main()
