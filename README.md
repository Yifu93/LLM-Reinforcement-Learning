# **QWEN\_FINETUNE**
---

<img src="./qwen_structure.png" alt="Qwen model structure" width="1000" />

### checkpoints/

- The `initial/` inital model as start point and reference
- `sft/, dpo/, rloo/` for intermediate checkpoints for each training stage
- `final` the fully fine-tuned weights

---

### data/

Dataset for fine-tuning.

```
data/
└── smoltalk/          # Supervised fine-tuning (SFT) pairs
└── ultrafeedback/     # RLHF sets → DPO/RLOO
└── WarmStart Dataset  # Supervised fine-tuning (SFT) for question-answers
└── Countdown_dataset  # TinyZero sets → Need Rule-Based Reward Function (DPO/RLOO)
```
where come from:
```
# Download the dataset from Hugging Face Hub
# -- Instrcution Following Dataset --
# (1) SmolTalk (Dataset for SFT): https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
# (2) UltraFeedback (Dataset for DPO and RLOO): https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
# -- Verifier-Based Dataset --
# (3) WarmStart (Dataset for SFT): https://huggingface.co/datasets/Asap7772/cog_behav_all_strategies
# (4) PromptsDataset from TinyZero (RLOO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
# (5) On-Policy Preference Dataset (the same as PromptsDataset, DPO): https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
```

**For training:**
- Preference Datasets:
    - smoltalk
    - ultrafeedback (training split)
- Verifier-Based Datasets: 
    - Warmstart
    - Countdown (training split)

**For evaluation:**
- Evaluating Ultrafeedback (test set)
- Evaluating Countdown (test set)

**Insights**

...

---



### Model

Define the model -- Qwen 2.5 0.5B

Key helpers

```python
def load_tokenizer(model_name, trust_remote_code=True):
    ...

def load_model(model_name, quant=None, lora_cfg=None):
    ...

def save_model(model, save_dir):
    ...
```

---
### Tokenizer

Special tokens:
```
151644: <|im_start|>

151645: <|im_end|>

151643: <|endoftext|> (Padding)

198: \n
```

Example messages:
```
<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n

<|im_start|>user\nPrompt text<|im_end|>\n

<|im_start|>assistant\nResponse text<|im_end|>
```

Masking Strategy:
```
labels[:len(prompt_ids)] = -100      # Ignore the prompt part in labels

labels[attention_mask == 0] = -100   # Ensure padding tokens are ignored
```

---
### Evaluation
```
evaluation.py
```
<img src="./qwen_project_evaluation.png" alt="Qwen model structure" width="1000" />

**outputs/** - to record the evaluation responses

**Generated inference** used for offline analysis.

---

### scripts/

| Script                  | What it does                                                                |
| ----------------------- | --------------------------------------------------------------------------- |
| `save_vanilla_model.py` | Utility to save the untouched base model into *checkpoints/initial/*.       |
| `data_download.py`      | Fetch and version datasets; pushes to *data/* in correct layout.            |
| `dataloader.py`         | Builds `torch.utils.data.Dataset` / `Dataset` objects from raw files.       |
| `train_model.py`        | Main training loop; picks config, stage (SFT, DPO, RLOO), handles resume.   |
| `evaluation.py`         | Runs automated eval; writes to *outputs/*.                                  |

---

### enviroment.txt
The module needed are listed here.