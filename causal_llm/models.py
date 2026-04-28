"""Model loading: HF auth, 4-bit NF4 for the instruct models."""

import os
import warnings
from dataclasses import dataclass

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


@dataclass
class LoadedModels:
    models: dict
    tokenizers: dict
    skipped: list
    active: dict  # subset of MODEL_CONFIGS for keys that actually loaded


def setup_hf_auth():
    """Pull HF_TOKEN from Kaggle Secrets / Colab userdata if not already set."""
    if os.environ.get('HF_TOKEN'):
        return True

    try:
        from kaggle_secrets import UserSecretsClient
        try:
            tok = UserSecretsClient().get_secret('HF_TOKEN')
            if tok:
                os.environ['HF_TOKEN'] = tok
                print('HF_TOKEN loaded from Kaggle Secrets.')
                return True
        except Exception as e:
            print(f'[INFO] Kaggle detected but HF_TOKEN secret not configured ({e}).')
    except ImportError:
        pass

    try:
        from google.colab import userdata
        try:
            tok = userdata.get('HF_TOKEN')
            if tok:
                os.environ['HF_TOKEN'] = tok
                print('HF_TOKEN loaded from Colab userdata.')
                return True
        except Exception:
            pass
    except ImportError:
        pass

    if os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        return True

    try:
        from huggingface_hub import HfFolder
        if HfFolder.get_token():
            return True
    except Exception:
        pass

    return False


def _count_params_with_tied_embeddings(model):
    """HF model cards count tied input/output embeddings twice; sum(p.numel())
    counts them once. Add the embedding back so our number matches the card.
    """
    n = sum(p.numel() for p in model.parameters())
    try:
        in_emb = model.get_input_embeddings()
        out_emb = model.get_output_embeddings()
        if (in_emb is not None and out_emb is not None
                and in_emb.weight is not None and out_emb.weight is not None
                and in_emb.weight.data_ptr() == out_emb.weight.data_ptr()):
            n += in_emb.weight.numel()
    except Exception:
        pass
    return n


def load_models(device, configs=None):
    configs = configs or MODEL_CONFIGS
    models, tokenizers, skipped = {}, {}, []

    for key, cfg in configs.items():
        print(f"\nLoading {cfg['display_name']} ({cfg['params']})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg['name'], trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if cfg['quantize'] and torch.cuda.is_available():
                model_config = AutoConfig.from_pretrained(cfg['name'], trust_remote_code=True)
                if getattr(model_config, 'pad_token_id', None) is None:
                    model_config.pad_token_id = tokenizer.eos_token_id

                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                # Pin to one GPU. device_map='auto' splits layers across GPUs
                # on multi-GPU boxes, which crashes scoring when input_ids on
                # one device hit an embedding on another.
                target = str(device) if device.type == 'cuda' else 'auto'
                model = AutoModelForCausalLM.from_pretrained(
                    cfg['name'],
                    config=model_config,
                    quantization_config=bnb,
                    device_map={'': target},
                    trust_remote_code=True,
                    attn_implementation='eager',
                )
            elif cfg['quantize']:
                print(f"  SKIPPING {cfg['display_name']}: 4-bit quant requires GPU.")
                skipped.append(key)
                continue
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg['name'],
                    trust_remote_code=True,
                    attn_implementation='eager',
                ).to(device)

            model.eval()
            models[key] = model
            tokenizers[key] = tokenizer

            n_params = _count_params_with_tied_embeddings(model)
            mem_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6
            print(f"  {cfg['display_name']}: {n_params:,} params, ~{mem_mb:.0f} MB")

        except Exception as e:
            print(f"  FAILED to load {cfg['display_name']}: {e}")
            skipped.append(key)

    active = {k: v for k, v in configs.items() if k in models}
    print(f"\nLoaded {len(models)}/{len(configs)} models on {device}.")
    if skipped:
        print(f"Skipped: {skipped}")
    if torch.cuda.is_available():
        print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return LoadedModels(models=models, tokenizers=tokenizers, skipped=skipped, active=active)
