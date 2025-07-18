import time
import torch
import pandas as pd
from transformers import AutoTokenizer, TrainerCallback
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM
from scripts.dataloader import get_ultrafeedback_dataset

# ──────────────────────────────────────────────────────────────
MODEL_PATH  = "checkpoints/merged_SmolTak"
DATA_PATH   = "./data/ultrafeedback_binarized/train_prefs"
OUTPUT_DIR  = "./checkpoints/DPO_ultrafeedback_lora"
MAX_LENGTH  = 1024
BATCH_SIZE  = 2
EPOCHS      = 3
# ──────────────────────────────────────────────────────────────


class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.log_history = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            step = state.global_step
            self.log_history.setdefault(step, {"step": step})
            self.log_history[step]["loss"] = logs["loss"]
            print(f"[Step {step}] Train Loss = {logs['loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_history:
            df = pd.DataFrame(list(self.log_history.values()))
            df.to_csv("loss_log_dpo_lora.csv", index=False)
            print("Training loss saved to loss_log_dpo_lora.csv")


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
    # ── 1. Load tokenizer and base model ─────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Base model loaded from {MODEL_PATH}")

    # Optionally set dropout on the base model (still useful under LoRA)
    if hasattr(base_model.config, "hidden_dropout_prob"):
        base_model.config.hidden_dropout_prob = 0.1
    if hasattr(base_model.config, "attention_probs_dropout_prob"):
        base_model.config.attention_probs_dropout_prob = 0.1

    # ── 2. Define LoRA configuration ──────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    # ── 3. Wrap the base model with LoRA adapters ─────────────────────────
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    print("LoRA adapters injected. Only LoRA parameters will be trained.")

    # ── 4. Load and format your pre‐tokenized DPO dataset ─────────────────
    train_dataset = get_ultrafeedback_dataset(DATA_PATH)
    if "prompt" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("prompt")

    # ── 5. Configure DPOTrainer hyperparameters ──────────────────────────
    dpo_config = DPOConfig(
        beta=0.1,
        max_length=MAX_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=EPOCHS,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # ── 6. Create and run DPOTrainer ─────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[PrintLossCallback(), SpeedCallback()],
    )

    print("Starting DPO+LoRA training…")
    trainer.train()

    # ── 7. Save only the LoRA adapters ──────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DPO+LoRA training complete. Adapters saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
