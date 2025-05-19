# models/qwen_model.py
# Qwen model loading and saving utilities

import logging
from pathlib import Path
from typing import Union, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if not already attached (prevents double logging)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"


# ──────────────────────────────────────────────────────────────
def _str_to_dtype(flag: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(flag.lower(), torch.float32)


def _prepare_tokenizer(tok: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Ensure tokenizer has a valid pad token and correct padding behavior."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ──────────────────────────────────────────────────────────────
def download_tokenizer(model_id: str = _DEFAULT_MODEL_ID) -> PreTrainedTokenizer:
    """Download tokenizer from Hugging Face Hub with safe config."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _prepare_tokenizer(tok)


def download_model(
    model_id: str = _DEFAULT_MODEL_ID,
    dtype: str = "bf16",
    device_map: str = "auto",
) -> PreTrainedModel:
    """Download model from Hugging Face Hub."""
    logger.info("Downloading model from %s (dtype=%s)", model_id, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_str_to_dtype(dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    return model


# ──────────────────────────────────────────────────────────────
def load_tokenizer(source: Union[str, Path] = _DEFAULT_MODEL_ID) -> PreTrainedTokenizer:
    """Load tokenizer from local directory or HF Hub and fix padding config."""
    tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    return _prepare_tokenizer(tok)


def load_model(
    model_id: str = _DEFAULT_MODEL_ID,
    checkpoint_path: Optional[Union[str, Path]] = None,
    dtype: str = "bf16",
    device_map: str = "auto",
) -> PreTrainedModel:
    """
    Load model from local checkpoint or HF Hub.
    If `checkpoint_path` is provided, that takes priority.
    """
    source = checkpoint_path or model_id
    logger.info("Loading model from %s (dtype=%s)", source, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=_str_to_dtype(dtype),
        device_map=device_map,
        trust_remote_code=True,
    )

    # Set pad_token_id to match tokenizer
    tok = load_tokenizer(source)
    model.config.pad_token_id = tok.pad_token_id

    return model


# ──────────────────────────────────────────────────────────────
def save_model_and_tokenizer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    save_dir: Union[str, Path],
):
    """Save both model and tokenizer to the given directory."""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model and tokenizer to %s", p)
    tokenizer.save_pretrained(p)
    model.save_pretrained(p)
    logger.info("✓ Model and tokenizer saved.")


def save_model(
    model: PreTrainedModel,
    save_dir: Union[str, Path],
):
    """Save both model and tokenizer to the given directory."""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model to %s", p)
    model.save_pretrained(p)
    logger.info("✓ Model saved.")
