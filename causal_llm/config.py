"""Run-wide configuration: models, paths, prompting strategies."""

from __future__ import annotations

from pathlib import Path

SEED = 42

MODEL_CONFIGS = {
    'gpt2_small': {
        'name': 'gpt2',
        'display_name': 'GPT-2 Small',
        'params': '124M',
        'quantize': False,
    },
    'gpt2_large': {
        'name': 'gpt2-large',
        'display_name': 'GPT-2 Large',
        'params': '774M',
        'quantize': False,
    },
    'qwen_1_5b': {
        'name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'display_name': 'Qwen2.5-1.5B-Instruct',
        'params': '1.5B',
        'quantize': True,
    },
    'llama_3_3b': {
        'name': 'meta-llama/Llama-3.2-3B-Instruct',
        'display_name': 'Llama-3.2-3B-Instruct',
        'params': '3.2B',
        'quantize': True,
    },
    'gemma_2_2b': {
        'name': 'google/gemma-2-2b-it',
        'display_name': 'Gemma-2-2B-it',
        'params': '2.6B',
        'quantize': True,
    },
}

MODEL_LABELS = {
    'gpt2_small':  'GPT-2 Small (124M)',
    'gpt2_large':  'GPT-2 Large (774M)',
    'qwen_1_5b':   'Qwen2.5-1.5B-Inst',
    'llama_3_3b':  'Llama-3.2-3B-Inst',
    'gemma_2_2b':  'Gemma-2-2B-it',
}

MODEL_COLORS = {
    'gpt2_small':  '#4C72B0',
    'gpt2_large':  '#DD8452',
    'qwen_1_5b':   '#8172B3',
    'llama_3_3b':  '#C44E52',
    'gemma_2_2b':  '#937860',
}

LEVEL_COLORS = {'L1': '#4C72B0', 'L2': '#DD8452', 'L3': '#C44E52'}

# Models that use raw-generation scoring rather than PMI; their accuracy
# numbers shouldn't be compared apples-to-apples with instruct models.
AT_CHANCE_MODELS = {'gpt2_small', 'gpt2_large'}

NUM_QUESTIONS_PER_CELL = 20  # per (graph_type, reasoning_level)

MAX_SEQ_LEN = 512
MODEL_MAX_SEQ_LEN = {
    'gpt2_small': 512,
    'gpt2_large': 512,
    'qwen_1_5b':  2048,
    'llama_3_3b': 2048,
    'gemma_2_2b': 2048,
}

PROMPTING_STRATEGIES = ['zero_shot', 'few_shot', 'chain_of_thought', 'causal_chain']
STRATEGY_LABELS = {
    'zero_shot': 'Zero-Shot',
    'few_shot': 'Few-Shot',
    'chain_of_thought': 'CoT',
    'causal_chain': 'Causal Chain',
}

OUTPUT_DIR = Path('./output_llm')
