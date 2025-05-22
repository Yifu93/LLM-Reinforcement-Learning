import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from models.qwen_model import load_tokenizer
from scripts.dataloader import get_warmstart_dataset  # 已 tokenized 的 warmstart 数据集

# Optional: Callback to log loss to terminal and file
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            loss_val = logs["loss"]
            print(f"Step {state.global_step}: loss = {loss_val:.4f}")
            with open("loss_log.csv", "a") as f:
                f.write(f"{state.global_step},{loss_val:.4f}\n")

def main():
    model_path = "./qwen2_model"

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)
    tokenizer.pad_token = tokenizer.eos_token  # Required for Qwen

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    train_dataset = get_warmstart_dataset("./data/warmstart/train")  # ensure correct subdir

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./sft_qwen_warmstart_model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        callbacks=[PrintLossCallback()],
    )

    # Train and save
    trainer.train()
    trainer.save_model("./sft_qwen_warmstart_model")

if __name__ == "__main__":
    main()