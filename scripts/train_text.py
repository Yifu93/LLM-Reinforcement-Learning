

# SFT training



# DPO training

from trl import DPOTrainer
from transformers import AutoModelForCausalLM
from datasets import Dataset

# Create a simple dataset with prompts, chosen, and rejected responses
dataset_dict = {
    "prompt": prompts,
    "chosen": chosen_responses,
    "rejected": rejected_responses
}

dataset = Dataset.from_dict(dataset_dict)

# Load base model
model = 

# Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # If None, the model itself is used as reference
    args=DPOTrainerArguments(
        per_device_train_batch_size=4,
        learning_rate=5e-7,
        max_length=512,
        output_dir="./dpo_model",
        num_train_epochs=3,
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
    beta=0.1,  # Hyperparameter for DPO loss
)

# Train the model
dpo_trainer.train()