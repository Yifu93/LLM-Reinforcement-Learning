import time
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from torch.utils.data import random_split
from scripts.dataloader import get_smoltalk_dataset
from transformers import DataCollatorForLanguageModeling

class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.log_history = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if step not in self.log_history:
                self.log_history[step] = {"step": step}

            record = self.log_history[step]
            if "loss" in logs:
                record["train_loss"] = logs["loss"]
                print(f"[Step {step}] Train Loss = {logs['loss']:.4f}")
            if "eval_loss" in logs:
                record["eval_loss"] = logs["eval_loss"]
                print(f"[Step {step}] Eval Loss = {logs['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_history:
            df = pd.DataFrame(list(self.log_history.values()))
            df.to_csv("loss_log.csv", index=False)
            print("Training and validation loss saved to loss_log.csv")

class SpeedCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time:
            duration = time.time() - self.start_time

            if state.global_step % 10 == 0:
                print(f"[Step {state.global_step}] Step Time: {duration:.2f}s")


def main():
    model_path = "./qwen2_model"
    dataset_path = "./data/smoltalk/train"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Add dropout
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = 0.1
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = 0.1

    # Load dataset - split into train/val
    full_dataset = get_smoltalk_dataset(dataset_path)
    train_size = int(0.95 * len(full_dataset))  # 95% for training
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use subset of validation set for speed
    val_subset_size = min(500, len(val_dataset))
    eval_dataset = torch.utils.data.Subset(val_dataset, list(range(val_subset_size)))

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")


    # Training arguments (no validation)
    training_args = TrainingArguments(
        output_dir="./sft_qwen_text_full",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=1,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=1,
        report_to="none",
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,  # Choose which dataset to use
        callbacks=[
            PrintLossCallback(),
            SpeedCallback(),
        ],
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model + tokenizer
    trainer.save_model("./checkpoints/sft_qwen_text_full")
    tokenizer.save_pretrained("./checkpoints/sft_qwen_text_full")
    print("Training complete and model saved!")


if __name__ == "__main__":
    main()

