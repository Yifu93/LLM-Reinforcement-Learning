from scripts.dataloader import get_smoltalk_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import logging
import torch
import time
from scripts.dataloader import get_smoltalk_dataloader

dataloader = get_smoltalk_dataloader(path="./data/smoltalk/train")
