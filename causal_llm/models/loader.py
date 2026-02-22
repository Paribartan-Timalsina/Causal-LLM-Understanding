"""Model loading utilities for GPT-2 and Phi-2 with optional quantization."""

import gc

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging,
)

from ..config import MODEL_CONFIGS, get_device

hf_logging.set_verbosity_error()


def get_available_models():
    """Return list of available model keys."""
    return list(MODEL_CONFIGS.keys())


def load_model(model_key, device=None):
    """Load a model and its tokenizer.

    Args:
        model_key: Key from MODEL_CONFIGS (e.g. 'gpt2_small', 'phi2').
        device: torch.device or None (auto-detected).

    Returns:
        tuple: (model, tokenizer)
    """
    if device is None:
        device = get_device()

    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )

    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["name"]
    quantize = cfg["quantize"]

    print(f"\nLoading {cfg['display_name']}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device)

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    mem_mb = param_count * 4 / 1e6
    print(f"  {cfg['display_name']}: {param_count:,} params, ~{mem_mb:.0f} MB")

    return model, tokenizer


def unload_model(model):
    """Free GPU memory occupied by a model."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
