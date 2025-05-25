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
from scripts.dataloader import get_warmstart_dataset

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
            print(f"[Step {state.global_step}] Step Time: {duration:.2f}s")

def main():
    model_path = "./qwen2_model"
    dataset_path = "./data/warmstart/train"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Add dropout (if supported)
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = 0.1
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = 0.1

    # Load dataset
    full_dataset = get_warmstart_dataset(dataset_path)
    train_size = int(0.99 * len(full_dataset))  # 99% for training
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset loaded: train = {len(train_dataset)}, val = {len(val_dataset)}")

    # Training settings
    training_args = TrainingArguments(
        output_dir="./sft_qwen_math",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        learning_rate=5e-6,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            PrintLossCallback(),
            SpeedCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model + tokenizer
    trainer.save_model("./checkpoints/sft_qwen_math_02")
    tokenizer.save_pretrained("./checkpoints/sft_qwen_math_02")
    print("Training complete and model saved!")


if __name__ == "__main__":
    main()