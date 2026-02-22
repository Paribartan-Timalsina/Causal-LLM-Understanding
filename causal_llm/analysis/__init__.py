__all__ = [
    "extract_hidden_states",
    "run_probing",
    "extract_attention",
    "analyze_causal_keyword_attention",
    "compute_confusion_matrices",
    "identify_failure_patterns",
]


def __getattr__(name):
    if name in ("extract_hidden_states", "run_probing"):
        from . import probing
        return getattr(probing, name)
    if name in ("extract_attention", "analyze_causal_keyword_attention"):
        from . import attention
        return getattr(attention, name)
    if name in ("compute_confusion_matrices", "identify_failure_patterns"):
        from . import error_analysis
        return getattr(error_analysis, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
