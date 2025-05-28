import sys
import time
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from scripts.dataloader import get_smoltalk_dataset  

class PrintLossCallback(TrainerCallback):
    def __init__(self):
        self.log_history = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if step not in self.log_history:
                self.log_history[step] = {"step": step}

            record = self.log_history[step]
            msg = f"[Step {step}]"

            if "loss" in logs:
                record["train_loss"] = logs["loss"]
                msg += f" Train Loss = {logs['loss']:.4f}"

            if "eval_loss" in logs:
                record["eval_loss"] = logs["eval_loss"]
                msg += f" Eval Loss = {logs['eval_loss']:.4f}"

            if state.epoch is not None:
                msg += f" | Epoch = {state.epoch:.2f}"

            msg += f" @ {time.strftime('%H:%M:%S')}"
            print(msg)

    def on_train_end(self, args, state, control, **kwargs):
        df = pd.DataFrame(list(self.log_history.values()))
        df.to_csv("training_and_validation_loss.csv", index=False)
        print("Loss history saved to training_and_validation_loss.csv")

class SpeedCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        duration = time.time() - self.start_time
        print(f"[Step {state.global_step}] Step Time: {duration:.2f}s")

def main():
    model_path = "./qwen2_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load SmolTalk dataset
    dataset = get_smoltalk_dataset("./data/smoltalk/train")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _full_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Use subset of validation set for speed
    val_subset_size = min(200, len(_full_val))
    eval_dataset = torch.utils.data.Subset(_full_val, list(range(val_subset_size)))

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Define callbacks
    loss_callback = PrintLossCallback()
    speed_callback = SpeedCallback()
    early_stop = EarlyStoppingCallback(early_stopping_patience=3)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_sft",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-6,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=4000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            loss_callback,
            speed_callback,
            early_stop,
        ],
    )

    print("Starting training...")
    trainer.train()

    model.save_pretrained("./lora_sft")
    tokenizer.save_pretrained("./lora_sft")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()