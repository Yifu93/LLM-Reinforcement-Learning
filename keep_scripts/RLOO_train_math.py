from scripts.dataloader import get_Countdown_dataset_RLOO

data_path = "data/Countdown-Tasks-3to4/train"
dataset = get_Countdown_dataset_RLOO(data_path)

def extract_math_prompts(raw: list[dict]) -> tuple[list[str], list[dict]]:
    prompts, meta = [], []
    for ex in raw:
        nums, tgt = ex["num"], ex["target"]
        prompt = (f"A conversation between User and Assistant. "
                  f"The user asks a question, and the Assistant solves it. "
                  f"The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                  f"User: Using the numbers {nums}, create an equation that equals {tgt}. "
                  f"You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                  f"Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, "
                  f"for example <answer> (1 + 2) / 3 </answer>. Assistant: Let me solve this step by step.")
        prompts.append(prompt)
        meta.append({"numbers": nums, "target": tgt})
    return prompts, meta


# import time
# import torch
# import pandas as pd
# from transformers import AutoTokenizer, TrainerCallback, AutoModelForCausalLM, AutoModelForSequenceClassification
# from trl import RLOOConfig, RLOOTrainer
# from scripts.dataloader import load_tokenizer, get_ultrafeedback_dataset_RLOO
# from functions.countdown import compute_score

# # ──────────────────────────────────────────────
# MODEL_PATH = "checkpoints/merged_SmolTak"
# DATA_PATH = "./data/ultrafeedback_binarized/train_gen"
# OUTPUT_DIR = "./checkpoints/RLOO_ultrafeedback"
# MAX_LENGTH = 1024
# BATCH_SIZE = 4
# NUM_GEN_PER_PROMPT = 4
# EPOCHS = 1
# # ──────────────────────────────────────────────

# class PrintLossCallback(TrainerCallback):
#     def __init__(self):
#         self.log_history = {}

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs:
#             step = state.global_step
#             if step not in self.log_history:
#                 self.log_history[step] = {"step": step}
#             if "loss" in logs:
#                 self.log_history[step]["loss"] = logs["loss"]
#                 print(f"[Step {step}] Train Loss = {logs['loss']:.4f}")

#     def on_train_end(self, args, state, control, **kwargs):
#         if self.log_history:
#             df = pd.DataFrame(list(self.log_history.values()))
#             df.to_csv("loss_log_rloo.csv", index=False)
#             print("Training loss saved to loss_log_rloo.csv")

# class SpeedCallback(TrainerCallback):
#     def __init__(self):
#         self.start_time = None

#     def on_step_begin(self, args, state, control, **kwargs):
#         self.start_time = time.time()

#     def on_step_end(self, args, state, control, **kwargs):
#         if self.start_time:
#             duration = time.time() - self.start_time
#             print(f"[Step {state.global_step}] Step Time: {duration:.2f}s")

# def wrap_reward_model(compute_score, reward_tokenizer, device):

#     def compute_reward(full_texts):  # ← list of full prompt+response strings
#         with torch.no_grad():
#             inputs = reward_tokenizer(
#                 full_texts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True
#             ).to(device)

#             outputs = [compute_score(text, )]
#             rewards = outputs.logits.squeeze(-1)
#             # print('rewards:', rewards)
#             return rewards.tolist()
        
#     return compute_reward


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     tokenizer = load_tokenizer(MODEL_PATH)
#     tokenizer.pad_token = tokenizer.eos_token

#     policy_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     policy_model.config.pad_token_id = tokenizer.pad_token_id

#     ref_policy_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
#     ref_policy_model.config.pad_token_id = tokenizer.pad_token_id

#     reward_model_raw = compute_score
#     reward_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     reward_model = wrap_reward_model(reward_model_raw, reward_tokenizer, device)

#     train_dataset = get_ultrafeedback_dataset_RLOO(DATA_PATH)
#     print(train_dataset)

#     rloo_config = RLOOConfig(
#         # max_length=MAX_LENGTH,
#         per_device_train_batch_size=BATCH_SIZE,
#         gradient_accumulation_steps=8,
#         learning_rate=5e-6,
#         num_train_epochs=EPOCHS,
#         save_steps=200,
#         save_total_limit=2,
#         logging_steps=50,
#         output_dir=OUTPUT_DIR,
#         remove_unused_columns=True,
#         fp16=torch.cuda.is_available(),
#         num_sample_generations = 0, # No evaluation during training
#         report_to="none",
#         num_ppo_epochs=EPOCHS,
#         rloo_k=NUM_GEN_PER_PROMPT,
#         kl_coef=0.03,
#     )

#     trainer = RLOOTrainer(
#         config=rloo_config,
#         policy=policy_model,
#         ref_policy=ref_policy_model,
#         reward_model=reward_model,  # callable: (prompt, response) → reward
#         train_dataset=train_dataset,
#         processing_class=tokenizer,
#         callbacks=[PrintLossCallback(), SpeedCallback()],
#     )

#     print("Starting RLOO training...")
#     trainer.train()
#     trainer.save_model(OUTPUT_DIR)
#     tokenizer.save_pretrained(OUTPUT_DIR)
#     print("RLOO training complete and model saved!")

# if __name__ == "__main__":
#     main()