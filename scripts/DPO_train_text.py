import time
import torch
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback
from trl import DPOTrainer, DPOConfig
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from scripts.dataloader import get_ultrafeedback_dataset

# ──────────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/merged_SmolTak"
DATA_PATH = "./data/ultrafeedback_binarized/train_prefs"
OUTPUT_DIR = "./checkpoints/DPO_ultrafeedback"
MAX_LENGTH = 1024
BATCH_SIZE = 8
EPOCHS = 3
# ──────────────────────────────────────────────────────────────


class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.log_history = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if step not in self.log_history:
                self.log_history[step] = {"step": step}
            if "loss" in logs:
                self.log_history[step]["loss"] = logs["loss"]
                print(f"[Step {step}] Train Loss = {logs['loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_history:
            df = pd.DataFrame(list(self.log_history.values()))
            df.to_csv("loss_log_dpo.csv", index=False)
            print("Training loss saved to loss_log_dpo.csv")


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
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded from {MODEL_PATH}")

    # Optionally set dropout
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = 0.1
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = 0.1

    # Load dataset
    train_dataset = get_ultrafeedback_dataset(DATA_PATH)

    # DPO config
    dpo_args = DPOConfig(
        beta=0.1,
        max_length=MAX_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        num_train_epochs=EPOCHS,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        output_dir="./DPO_ultrafeedback",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[
            PrintLossCallback(),
            SpeedCallback(),
        ],
    )

    # Train
    print("Starting DPO training...")
    trainer.train()

    # Save final model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DPO training complete and model saved!")


if __name__ == "__main__":
    main()
