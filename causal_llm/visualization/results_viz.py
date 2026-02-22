"""Results visualization: accuracy heatmaps, strategy comparisons, scaling curves."""

from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..config import (
    MODEL_CONFIGS, MODEL_LABELS, MODEL_COLORS, LEVEL_COLORS,
    PROMPTING_STRATEGIES, OUTPUT_DIR,
)


def plot_accuracy_heatmap(all_results, output_dir=None):
    """Plot accuracy heatmap by graph type and reasoning level for each model.

    Args:
        all_results: dict from run_full_evaluation (keyed by (model_key, strategy))
        output_dir: Directory to save figure.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_keys = [k for k in MODEL_CONFIGS if any(k == mk for mk, _ in all_results)]
    graph_types = ["chain", "fork", "collider", "diamond", "instrument"]
    levels = ["L1", "L2", "L3"]

    fig, axes = plt.subplots(1, len(model_keys), figsize=(6 * len(model_keys), 5))
    if len(model_keys) == 1:
        axes = [axes]

    for ax_idx, model_key in enumerate(model_keys):
        key = (model_key, "zero_shot")
        if key not in all_results:
            continue
        results = all_results[key]["results"]

        acc_matrix = np.zeros((len(graph_types), len(levels)))
        for gi, g in enumerate(graph_types):
            for li, l in enumerate(levels):
                matching = [r for r in results if r["graph"] == g and r["level"] == l]
                if matching:
                    acc_matrix[gi, li] = np.mean([r["correct"] for r in matching])

        sns.heatmap(acc_matrix, annot=True, fmt=".0%", cmap="RdYlGn",
                    xticklabels=levels, yticklabels=graph_types,
                    vmin=0, vmax=1, ax=axes[ax_idx])
        axes[ax_idx].set_title(MODEL_LABELS.get(model_key, model_key), fontweight="bold")

    plt.suptitle("Accuracy by Graph Type and Reasoning Level", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_graph_level.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_strategy_comparison(strategy_results, output_dir=None):
    """Plot prompting strategy comparison across models.

    Args:
        strategy_results: dict (model_key, strategy) -> {'accuracy': float, ...}
        output_dir: Directory to save figure.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    strategy_labels = {
        "zero_shot": "Zero-Shot",
        "few_shot": "Few-Shot",
        "chain_of_thought": "CoT",
        "causal_chain": "Causal Chain",
    }

    model_keys = sorted(set(mk for mk, _ in strategy_results))
    strategies = PROMPTING_STRATEGIES

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(strategies))
    width = 0.8 / len(model_keys)

    for i, model_key in enumerate(model_keys):
        accs = [strategy_results.get((model_key, s), {}).get("accuracy", 0)
                for s in strategies]
        axes[0].bar(x + i * width, accs, width,
                    label=MODEL_LABELS.get(model_key, model_key),
                    color=MODEL_COLORS.get(model_key, "#999999"), alpha=0.85)

    axes[0].set_title("Overall Accuracy by Prompting Strategy", fontweight="bold")
    axes[0].set_xticks(x + width * (len(model_keys) - 1) / 2)
    axes[0].set_xticklabels([strategy_labels.get(s, s) for s in strategies])
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 0.6)

    l2_data = defaultdict(dict)
    for (mk, s), data in strategy_results.items():
        results = data.get("results", [])
        l2_results = [r for r in results if r["level"] == "L2"]
        if l2_results:
            l2_data[mk][s] = np.mean([r["correct"] for r in l2_results])

    for i, model_key in enumerate(model_keys):
        accs = [l2_data.get(model_key, {}).get(s, 0) for s in strategies]
        axes[1].bar(x + i * width, accs, width,
                    label=MODEL_LABELS.get(model_key, model_key),
                    color=MODEL_COLORS.get(model_key, "#999999"), alpha=0.85)

    axes[1].set_title("L2 (Intervention) Accuracy by Strategy", fontweight="bold")
    axes[1].set_xticks(x + width * (len(model_keys) - 1) / 2)
    axes[1].set_xticklabels([strategy_labels.get(s, s) for s in strategies])
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_ylim(0, 0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "prompting_strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
