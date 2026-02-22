"""Central configuration for the Causal LLM Reasoning evaluation framework."""

import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "gpt2_small": {
        "name": "gpt2",
        "display_name": "GPT-2 Small (124M)",
        "params": "124M",
        "quantize": False,
    },
    "gpt2_large": {
        "name": "gpt2-large",
        "display_name": "GPT-2 Large (774M)",
        "params": "774M",
        "quantize": False,
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "display_name": "Phi-2 (2.7B)",
        "params": "2.7B",
        "quantize": True,
    },
}

MODEL_LABELS = {k: v["display_name"] for k, v in MODEL_CONFIGS.items()}

MAX_SEQ_LEN = 512
MODEL_MAX_SEQ_LEN = {
    "gpt2_small": 512,
    "gpt2_large": 512,
    "phi2": 2048,
}

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
NUM_QUESTIONS_PER_CELL = 20

PROMPTING_STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought", "causal_chain"]

# ---------------------------------------------------------------------------
# Probing settings
# ---------------------------------------------------------------------------
PROBE_NUM_SAMPLES = 200
PROBE_CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./output_llm")

# ---------------------------------------------------------------------------
# Visualization colors
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "gpt2_small": "#4C72B0",
    "gpt2_large": "#DD8452",
    "phi2": "#55A868",
}

LEVEL_COLORS = {
    "L1": "#4C72B0",
    "L2": "#DD8452",
    "L3": "#C44E52",
}


# ---------------------------------------------------------------------------
# Seed utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device (CUDA > CPU)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# YAML config loading (optional override)
# ---------------------------------------------------------------------------
def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file and return its contents as a dict."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
