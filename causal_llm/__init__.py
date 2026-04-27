"""Causal LLM evaluation: probing causal reasoning across Pearl's hierarchy."""

from .benchmark import benchmark_summary, build_benchmark, make_question
from .config import (
    AT_CHANCE_MODELS,
    LEVEL_COLORS,
    MAX_SEQ_LEN,
    MODEL_COLORS,
    MODEL_CONFIGS,
    MODEL_LABELS,
    MODEL_MAX_SEQ_LEN,
    NUM_QUESTIONS_PER_CELL,
    OUTPUT_DIR,
    PROMPTING_STRATEGIES,
    SEED,
    STRATEGY_LABELS,
)
from .evaluator import evaluate_model
from .graphs import CAUSAL_GRAPHS, plot_causal_graphs
from .models import LoadedModels, load_models, setup_hf_auth
from .prompts import CONTENT_FREE_PROMPTS, PROMPT_FORMATTERS
from .reporting import save_results
from .scenarios import SCENARIOS
from .scoring import (
    compute_calibration_priors,
    score_answer_generation,
    score_answer_logprob,
)
from .visualization import (
    plot_accuracy_heatmap,
    plot_benchmark_stats,
    plot_strategy_comparison,
)

__all__ = [
    # config
    'MODEL_CONFIGS', 'MODEL_LABELS', 'MODEL_COLORS', 'LEVEL_COLORS',
    'PROMPTING_STRATEGIES', 'STRATEGY_LABELS', 'OUTPUT_DIR', 'SEED',
    'MAX_SEQ_LEN', 'MODEL_MAX_SEQ_LEN', 'NUM_QUESTIONS_PER_CELL',
    'AT_CHANCE_MODELS',
    # graphs / scenarios / benchmark
    'CAUSAL_GRAPHS', 'SCENARIOS', 'plot_causal_graphs',
    'build_benchmark', 'benchmark_summary', 'make_question',
    # models / prompts / scoring / evaluator
    'load_models', 'LoadedModels', 'setup_hf_auth',
    'PROMPT_FORMATTERS', 'CONTENT_FREE_PROMPTS',
    'score_answer_logprob', 'score_answer_generation', 'compute_calibration_priors',
    'evaluate_model',
    # viz / reporting
    'plot_benchmark_stats', 'plot_accuracy_heatmap', 'plot_strategy_comparison',
    'save_results',
]
