__all__ = [
    "plot_benchmark_stats",
    "plot_causal_graphs",
    "plot_accuracy_heatmap",
    "plot_strategy_comparison",
    "plot_results_dashboard",
    "plot_confusion_matrices",
]


def __getattr__(name):
    if name in ("plot_benchmark_stats", "plot_causal_graphs"):
        from . import benchmark_viz
        return getattr(benchmark_viz, name)
    if name in ("plot_accuracy_heatmap", "plot_strategy_comparison"):
        from . import results_viz
        return getattr(results_viz, name)
    if name in ("plot_results_dashboard", "plot_confusion_matrices"):
        from . import dashboard
        return getattr(dashboard, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
