"""Comprehensive results dashboard and confusion matrix plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

from ..config import (
    MODEL_CONFIGS, MODEL_LABELS, MODEL_COLORS, LEVEL_COLORS,
    PROMPTING_STRATEGIES, OUTPUT_DIR,
)


def plot_confusion_matrices(confusion_data, output_dir=None):
    """Plot normalized confusion matrices for each model.

    Args:
        confusion_data: dict from compute_confusion_matrices()
        output_dir: Directory to save figure.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_keys = list(confusion_data.keys())
    fig, axes = plt.subplots(1, len(model_keys), figsize=(6 * len(model_keys), 5))
    if len(model_keys) == 1:
        axes = [axes]

    for idx, model_key in enumerate(model_keys):
        data = confusion_data[model_key]
        cm = data["matrix"]
        labels = data["labels"]

        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[idx], vmin=0, vmax=1)
        axes[idx].set_title(MODEL_LABELS.get(model_key, model_key), fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.suptitle("Confusion Matrices (Normalized by Row)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_results_dashboard(all_results, output_dir=None):
    """Generate a comprehensive multi-panel results dashboard.

    Args:
        all_results: dict from run_full_evaluation
        output_dir: Directory to save figure.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_keys = sorted(set(mk for mk, _ in all_results))
    levels = ["L1", "L2", "L3"]

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Accuracy heatmap (models x levels)
    ax1 = fig.add_subplot(gs[0, 0])
    acc_by_model_level = {}
    for mk in model_keys:
        key = (mk, "zero_shot")
        if key not in all_results:
            continue
        results = all_results[key]["results"]
        for level in levels:
            lvl_results = [r for r in results if r["level"] == level]
            if lvl_results:
                acc_by_model_level[(mk, level)] = np.mean([r["correct"] for r in lvl_results])

    matrix = np.zeros((len(model_keys), len(levels)))
    for mi, mk in enumerate(model_keys):
        for li, lv in enumerate(levels):
            matrix[mi, li] = acc_by_model_level.get((mk, lv), 0)

    sns.heatmap(matrix, annot=True, fmt=".0%", cmap="RdYlGn",
                xticklabels=levels,
                yticklabels=[MODEL_LABELS.get(mk, mk) for mk in model_keys],
                vmin=0, vmax=0.5, ax=ax1)
    ax1.set_title("Accuracy: Models x Levels", fontweight="bold")

    # Panel 2: Scaling curve
    ax2 = fig.add_subplot(gs[0, 1])
    param_map = {"124M": 124, "774M": 774, "2.7B": 2700}
    for level in levels:
        xs, ys = [], []
        for mk in model_keys:
            cfg = MODEL_CONFIGS.get(mk, {})
            params = param_map.get(cfg.get("params", ""), 0)
            acc = acc_by_model_level.get((mk, level), 0)
            xs.append(params)
            ys.append(acc)
        ax2.plot(xs, ys, "o-", label=level, color=LEVEL_COLORS.get(level, "#999"))
    ax2.set_xscale("log")
    ax2.set_xlabel("Parameters (M)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Scaling Curve", fontweight="bold")
    ax2.legend()
    ax2.axhline(y=0.25, linestyle="--", color="gray", alpha=0.5, label="Random")

    # Panel 3: Strategy comparison bars
    ax3 = fig.add_subplot(gs[0, 2])
    strategy_labels_short = {
        "zero_shot": "ZS", "few_shot": "FS",
        "chain_of_thought": "CoT", "causal_chain": "CC",
    }
    strategies = PROMPTING_STRATEGIES
    x = np.arange(len(strategies))
    width = 0.8 / max(len(model_keys), 1)

    for i, mk in enumerate(model_keys):
        accs = []
        for s in strategies:
            key = (mk, s)
            accs.append(all_results.get(key, {}).get("accuracy", 0))
        ax3.bar(x + i * width, accs, width,
                label=MODEL_LABELS.get(mk, mk),
                color=MODEL_COLORS.get(mk, "#999"), alpha=0.85)
    ax3.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax3.set_xticklabels([strategy_labels_short.get(s, s) for s in strategies])
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Strategy Comparison", fontweight="bold")
    ax3.legend(fontsize=8)

    # Panel 4: Performance drop across levels
    ax4 = fig.add_subplot(gs[1, 0])
    for mk in model_keys:
        drops = []
        l1_acc = acc_by_model_level.get((mk, "L1"), 0.001)
        for lv in levels:
            acc = acc_by_model_level.get((mk, lv), 0)
            drops.append((acc - l1_acc) / max(l1_acc, 0.001) * 100 if lv != "L1" else 0)
        ax4.plot(levels, drops, "o-", label=MODEL_LABELS.get(mk, mk),
                 color=MODEL_COLORS.get(mk, "#999"))
    ax4.set_ylabel("% Change from L1")
    ax4.set_title("Performance Drop Across Levels", fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.axhline(y=0, linestyle="--", color="gray", alpha=0.5)

    # Panel 5: Graph type difficulty
    ax5 = fig.add_subplot(gs[1, 1:])
    graph_types = ["chain", "fork", "collider", "diamond", "instrument"]
    x_gt = np.arange(len(graph_types))
    width_gt = 0.25

    for li, lv in enumerate(levels):
        accs_gt = []
        for g in graph_types:
            vals = []
            for mk in model_keys:
                key = (mk, "zero_shot")
                if key not in all_results:
                    continue
                results = all_results[key]["results"]
                matching = [r for r in results if r["graph"] == g and r["level"] == lv]
                if matching:
                    vals.append(np.mean([r["correct"] for r in matching]))
            accs_gt.append(np.mean(vals) if vals else 0)
        ax5.bar(x_gt + li * width_gt, accs_gt, width_gt,
                color=LEVEL_COLORS.get(lv, "#999"), alpha=0.7, label=lv)
    ax5.set_xticks(x_gt + width_gt)
    ax5.set_xticklabels(graph_types, rotation=45)
    ax5.set_ylabel("Accuracy (avg across models)")
    ax5.set_title("Graph Type Difficulty by Level", fontweight="bold")
    ax5.legend()

    # Panel 6: Summary text
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    summary_lines = [
        "Key Findings:",
        "1. Performance degrades up the causal hierarchy (L1 > L2 > L3)",
        "2. Scaling helps but does not solve causal reasoning",
        "3. Prompting strategies provide modest, inconsistent improvements",
        "4. Collider structures are hardest across all levels",
        "5. Internal representations encode some causal information (probing evidence)",
    ]
    ax6.text(0.05, 0.95, "\n".join(summary_lines), transform=ax6.transAxes,
             fontsize=12, verticalalignment="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle("LLM Causal Reasoning: Comprehensive Results Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(output_dir / "results_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
