import sys
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from scripts.dataloader import get_smoltalk_dataset  

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"Step {state.global_step}: loss = {logs['loss']:.4f}")

class SpeedCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        duration = time.time() - self.start_time
        print(f"[Step {state.global_step}] Time: {duration:.2f}s")

def main():
    model_path = "./qwen2_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

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

    train_dataset = get_smoltalk_dataset("./data/smoltalk/train")

    training_args = TrainingArguments(
        output_dir="./sft",
        per_device_train_batch_size=16,
        num_train_epochs=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=4000,
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=2,  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        callbacks=[
            PrintLossCallback(),
            SpeedCallback(),  
        ],
    )

    trainer.train()
    model.save_pretrained("./sft")
    tokenizer.save_pretrained("./sft")

if __name__ == "__main__":
    main()