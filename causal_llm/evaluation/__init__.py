from .prompts import (
    format_zero_shot,
    format_few_shot,
    format_chain_of_thought,
    format_causal_chain,
    PROMPT_FORMATTERS,
)

__all__ = [
    "format_zero_shot",
    "format_few_shot",
    "format_chain_of_thought",
    "format_causal_chain",
    "PROMPT_FORMATTERS",
    "score_answer_logprob",
    "score_answer_generation",
    "compute_calibration_priors",
    "evaluate_model",
    "run_full_evaluation",
]


def __getattr__(name):
    if name in ("score_answer_logprob", "score_answer_generation", "compute_calibration_priors"):
        from . import scoring
        return getattr(scoring, name)
    if name in ("evaluate_model", "run_full_evaluation"):
        from . import evaluator
        return getattr(evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
