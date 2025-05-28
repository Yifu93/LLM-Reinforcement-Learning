import time
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from torch.utils.data import ConcatDataset
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

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_history:
            df = pd.DataFrame(list(self.log_history.values()))
            df.to_csv("loss_log.csv", index=False)
            print("Training loss saved to loss_log.csv")


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
    model_path = "checkpoints/initial"
    dataset_path = "./data/warmstart/train"
    dataset_test_path = "./data/warmstart/test"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded from {model_path}")

    # Set dropout if supported
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = 0.1
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = 0.1

    # Load dataset (no validation split)
    train_dataset = get_warmstart_dataset(dataset_path)
    test_dataset = get_warmstart_dataset(dataset_test_path)
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    if train_dataset is None or test_dataset is None:
        raise RuntimeError("Failed to load one or both datasets!")


    # Training arguments (no validation)
    training_args = TrainingArguments(
        output_dir="./sft_qwen_math",
        per_device_train_batch_size=16,
        learning_rate=5e-6,
        num_train_epochs=10,
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
        train_dataset=combined_dataset,  # Choose which dataset to use
        callbacks=[
            PrintLossCallback(),
            SpeedCallback(),
        ],
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save model and tokenizer
    trainer.save_model("./checkpoints/SFT_qwen_math_03")
    tokenizer.save_pretrained("./checkpoints/SFT_qwen_math_03")
    print("Training complete and model saved!")


if __name__ == "__main__":
    main()
