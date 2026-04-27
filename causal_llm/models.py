"""Model loading: HF authentication, 4-bit quantization, tied-aware param counts."""

from __future__ import annotations

import os
import warnings
from typing import NamedTuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as hf_logging,
)

from .config import MODEL_CONFIGS

hf_logging.set_verbosity_error()
warnings.filterwarnings('ignore')


def setup_hf_auth() -> bool:
    """Pull an HF token from Kaggle Secrets / Colab userdata / env into HF_TOKEN.

    Returns True if a token is available (env or cached login). License-gated
    models (Llama, Gemma) require this to load.
    """
    if not os.environ.get('HF_TOKEN'):
        try:
            from kaggle_secrets import UserSecretsClient
            try:
                tok = UserSecretsClient().get_secret('HF_TOKEN')
                if tok:
                    os.environ['HF_TOKEN'] = tok
                    print('HF_TOKEN loaded from Kaggle Secrets.')
            except Exception as e:
                print(f'[INFO] Kaggle detected but HF_TOKEN secret not configured ({e}).')
        except ImportError:
            pass

    if not os.environ.get('HF_TOKEN'):
        try:
            from google.colab import userdata
            try:
                tok = userdata.get('HF_TOKEN')
                if tok:
                    os.environ['HF_TOKEN'] = tok
                    print('HF_TOKEN loaded from Colab userdata.')
            except Exception:
                pass
        except ImportError:
            pass

    has_token = bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'))
    if not has_token:
        try:
            from huggingface_hub import HfFolder
            has_token = bool(HfFolder.get_token())
        except Exception:
            pass
    return has_token


class LoadedModels(NamedTuple):
    models: dict
    tokenizers: dict
    skipped: list
    active: dict  # subset of MODEL_CONFIGS for keys that actually loaded


def _count_params_including_tied(model) -> int:
    """Param count that includes tied input/output embeddings.

    `sum(p.numel() for p in model.parameters())` counts a tied embedding once,
    while HF "nominal" model-card counts include it twice. This restores the
    nominal figure so reported sizes match the model cards.
    """
    base = sum(p.numel() for p in model.parameters())
    try:
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        if (in_emb is not None and out_emb is not None
                and getattr(in_emb, 'weight', None) is not None
                and getattr(out_emb, 'weight', None) is not None
                and in_emb.weight.data_ptr() == out_emb.weight.data_ptr()):
            base += in_emb.weight.numel()
    except Exception:
        pass
    return base


def load_models(
    device: torch.device,
    configs: dict | None = None,
) -> LoadedModels:
    """Load every configured model; skip on failure (license / OOM / etc.)."""
    configs = configs or MODEL_CONFIGS
    models: dict = {}
    tokenizers: dict = {}
    skipped: list = []

    for key, cfg in configs.items():
        print(f"\nLoading {cfg['display_name']} ({cfg['params']})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg['name'], trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if cfg['quantize'] and torch.cuda.is_available():
                # Pin to a single GPU so device_map='auto' doesn't split layers
                # across devices (which crashes scoring when input_ids meet an
                # embedding on another GPU).
                model_config = AutoConfig.from_pretrained(cfg['name'], trust_remote_code=True)
                if getattr(model_config, 'pad_token_id', None) is None:
                    model_config.pad_token_id = tokenizer.eos_token_id
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                target = str(device) if device.type == 'cuda' else 'auto'
                model = AutoModelForCausalLM.from_pretrained(
                    cfg['name'], config=model_config, quantization_config=bnb_config,
                    device_map={'': target}, trust_remote_code=True,
                    attn_implementation='eager',
                )
            elif cfg['quantize'] and not torch.cuda.is_available():
                print(f"  SKIPPING {cfg['display_name']}: 4-bit quantization requires GPU.")
                skipped.append(key)
                continue
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg['name'], trust_remote_code=True,
                    attn_implementation='eager',
                ).to(device)

            model.eval()
            models[key] = model
            tokenizers[key] = tokenizer

            n_params = _count_params_including_tied(model)
            mem_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6
            print(f"  {cfg['display_name']}: {n_params:,} params (tied-aware), ~{mem_mb:.0f} MB")

        except Exception as e:
            print(f"  FAILED to load {cfg['display_name']}: {e}")
            skipped.append(key)

    active = {k: v for k, v in configs.items() if k in models}
    print(f"\nLoaded {len(models)} / {len(configs)} models on {device}.")
    if skipped:
        print(f"Skipped: {skipped}")
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return LoadedModels(models=models, tokenizers=tokenizers, skipped=skipped, active=active)
