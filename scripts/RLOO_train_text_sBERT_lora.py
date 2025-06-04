import time
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    TrainerCallback,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import RLOOConfig, RLOOTrainer
from peft import get_peft_model, LoraConfig, TaskType
from scripts.dataloader import load_tokenizer, get_ultrafeedback_dataset_RLOO

# ──────────────────────────────────────────────────────────────
MODEL_PATH           = "checkpoints/merged_SmolTak"
REWARD_MODEL_PATH    = "./checkpoints/reward/BERT_sia"
DATA_PATH            = "./data/ultrafeedback_binarized/train_gen"
OUTPUT_DIR           = "./checkpoints/RLOO_ultrafeedback_lora_BERT_sia"
MAX_LENGTH           = 512
BATCH_SIZE           = 4
NUM_GEN_PER_PROMPT   = 4
EPOCHS               = 2
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
            df.to_csv("loss_log_rloo_lora.csv", index=False)
            print("Training loss saved to loss_log_rloo_lora.csv")


class SpeedCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"[Step {state.global_step}] Step Time: {duration:.2f}s")


def wrap_reward_model(reward_model, reward_tokenizer, device):
    reward_model.to(device)
    reward_model.eval()

    def compute_reward(full_texts):  # list of "prompt + response" strings
        with torch.no_grad():
            inputs = reward_tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(device)
            outputs = reward_model(**inputs)
            # Assume the classifier returns a single logit per sequence
            rewards = outputs.logits.squeeze(-1)
            return rewards.tolist()

    return compute_reward


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load tokenizer for policy and reference policy ───────────────
    tokenizer = load_tokenizer(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load and wrap policy model with LoRA ─────────────────────────
    base_policy = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    base_policy.config.pad_token_id = tokenizer.pad_token_id

    # Optional dropout tweaks on base (still apply under LoRA)
    if hasattr(base_policy.config, "hidden_dropout_prob"):
        base_policy.config.hidden_dropout_prob = 0.1
    if hasattr(base_policy.config, "attention_probs_dropout_prob"):
        base_policy.config.attention_probs_dropout_prob = 0.1

    # Define LoRA config (only these adapters are trainable)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    policy_model = get_peft_model(base_policy, lora_config)
    policy_model.print_trainable_parameters()
    print("LoRA adapters injected into policy. Only LoRA params will be trained.")

    policy_model.to(device)

    # ── 3. Load reference (frozen) policy ────────────────────────────────
    ref_policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    ref_policy_model.config.pad_token_id = tokenizer.pad_token_id
    ref_policy_model.to(device)
    ref_policy_model.eval()  # never train the reference policy

    # ── 4. Load and wrap reward model ────────────────────────────────────
    raw_reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    compute_reward = wrap_reward_model(raw_reward_model, reward_tokenizer, device)

    # ── 5. Load RLOO dataset ──────────────────────────────────────────────
    train_dataset = get_ultrafeedback_dataset_RLOO(DATA_PATH)
    # Dataset should have a "prompt" column of strings
    # and RLOOTrainer will tokenize on the fly via `tokenizer`.

    # ── 6. Configure RLOOTrainer ─────────────────────────────────────────
    rloo_config = RLOOConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=EPOCHS,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,  # keep "prompt" for on-the-fly tokenization
        fp16=torch.cuda.is_available(),
        num_sample_generations=0,      # no eval during training
        report_to="none",
        num_ppo_epochs=1,              # RLOO uses a single PPO epoch per batch
        rloo_k=NUM_GEN_PER_PROMPT,     # K completions per prompt
        kl_coef=0.03,                  # KL penalty weight
    )

    trainer = RLOOTrainer(
        config=rloo_config,
        policy=policy_model,
        ref_policy=ref_policy_model,
        reward_model=compute_reward,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[PrintLossCallback(), SpeedCallback()],
    )

    # ── 7. Train ─────────────────────────────────────────────────────────
    print("Starting RLOO+LoRA training…")
    trainer.train()

    # ── 8. Save only LoRA adapters and tokenizer ─────────────────────────
    policy_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("RLOO+LoRA training complete. Adapters saved at:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
