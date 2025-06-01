import time
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, TrainerCallback, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RLOOConfig, RLOOTrainer
from datasets import load_from_disk
from scripts.dataloader import load_tokenizer
from scripts.dataloader import get_rloo_dataset

# ──────────────────────────────────────────────
MODEL_PATH = "checkpoints/merged_SmolTak"
REWARD_MODEL_PATH = "OpenAssistant/reward-model-deberta-v3-large"
DATA_PATH = "./data/ultrafeedback_binarized/train_prompts"
OUTPUT_DIR = "./checkpoints/RLOO_ultrafeedback"
MAX_LENGTH = 1024
BATCH_SIZE = 2
EPOCHS = 1
# ──────────────────────────────────────────────

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
            df.to_csv("loss_log_rloo.csv", index=False)
            print("Training loss saved to loss_log_rloo.csv")

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

    def compute_reward(texts):
        with torch.no_grad():
            inputs = reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = reward_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            rewards = probs[:, 1]  # positive class probability
        return rewards

    return compute_reward

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    policy_model.config.pad_token_id = tokenizer.pad_token_id

    # Load reward model and wrap it
    reward_model_raw = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    reward_model = wrap_reward_model(reward_model_raw, reward_tokenizer, device)

    train_dataset = get_rloo_dataset(DATA_PATH)

    rloo_config = RLOOConfig(
        max_length=MAX_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=EPOCHS,
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        num_ppo_epochs=1,
        rloo_k=4,
        kl_coef=0.03,
    )

    trainer = RLOOTrainer(
        config=rloo_config,
        policy=policy_model,
        ref_policy=policy_model,
        reward_model=reward_model,  
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[PrintLossCallback(), SpeedCallback()],
    )

    print("Starting RLOO training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("RLOO training complete and model saved!")

if __name__ == "__main__":
    main()