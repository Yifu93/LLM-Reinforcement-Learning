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



### models/

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

### outputs/

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

### README.md

---

### requirements.txt