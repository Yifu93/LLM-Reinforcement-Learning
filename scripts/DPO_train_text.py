import re
from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from scripts.dataloader import get_ultrafeedback_dataset

# ──────────────────────────────────────────────────────────────
MAX_LENGTH    = 1024 
BATCH_SIZE    = 16
MODEL_PATH    = "checkpoints/merged_SmolTak"
DEVICE        = "cuda"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# ──────────────────────────────────────────────

# Load dataset (DPO)
data_path = "./data/ultrafeedback_binarized/train_prefs"
train_dataset = get_ultrafeedback_dataset(data_path)

