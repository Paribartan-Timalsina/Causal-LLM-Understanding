"""Plotting utilities for the benchmark and evaluation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from .config import (
    AT_CHANCE_MODELS,
    LEVEL_COLORS,
    MODEL_COLORS,
    MODEL_LABELS,
    PROMPTING_STRATEGIES,
    STRATEGY_LABELS,
)
from .graphs import CAUSAL_GRAPHS


def plot_benchmark_stats(benchmark: list[dict], output_path: Path | None = None) -> Figure:
    """Three-panel: count by graph×level, level pie, answer distribution."""
    df = pd.DataFrame(benchmark)
    ct = pd.crosstab(df['graph'], df['level'])[['L1', 'L2', 'L3']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ct.plot(kind='bar', ax=axes[0],
            color=[LEVEL_COLORS[l] for l in ['L1', 'L2', 'L3']])
    axes[0].set_title('Questions by Graph Type & Level', fontweight='bold')
    axes[0].set_xlabel('Causal Graph Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Level')

    level_counts = df['level'].value_counts().sort_index()
    level_labels = [str(l) for l in level_counts.index]
    axes[1].pie(level_counts, labels=level_labels,
                colors=[LEVEL_COLORS[l] for l in level_labels],
                autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Distribution by Reasoning Level', fontweight='bold')

    ans = df['answer'].value_counts().sort_index()
    axes[2].bar(list(ans.index), list(ans.values), color='#7FB3D8')
    axes[2].set_title('Answer Key Distribution', fontweight='bold')
    axes[2].set_xlabel('Correct Answer')
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_accuracy_heatmap(
    detailed_acc: dict,
    active_models: dict,
    output_path: Path | None = None,
) -> Figure:
    """One heatmap per model: rows = levels, cols = graph types."""
    n = len(active_models)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    graph_types = list(CAUSAL_GRAPHS.keys())
    levels = ['L1', 'L2', 'L3']

    for ax, model_key in zip(axes, active_models):
        data = np.zeros((len(levels), len(graph_types)))
        for i, lv in enumerate(levels):
            for j, gt in enumerate(graph_types):
                data[i, j] = detailed_acc[model_key].get(lv, {}).get(gt, 0)
        sns.heatmap(
            data, annot=True, fmt='.0%', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=graph_types, yticklabels=levels,
            ax=ax, cbar_kws={'shrink': 0.8},
        )
        ax.set_title(MODEL_LABELS[model_key], fontweight='bold')
        ax.set_xlabel('Causal Graph Type')

    axes[0].set_ylabel('Reasoning Level')
    plt.suptitle('Accuracy by Graph Type and Reasoning Level (Zero-Shot)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.text(0.01, -0.04,
             'GPT-2 models use raw-generation scoring (large prior range); '
             'instruct models use PMI-debiased scoring. Cross-model bars are not '
             'directly comparable.',
             fontsize=8, color='#555555')
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_strategy_comparison(
    strategy_results: dict,
    active_models: dict,
    output_path: Path | None = None,
) -> Figure:
    """Two-panel bar chart: overall acc, then L2-only acc, by strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(PROMPTING_STRATEGIES))
    width = 0.25

    def _draw(ax, getter, title):
        for i, mk in enumerate(active_models):
            vals = [getter(mk, s) for s in PROMPTING_STRATEGIES]
            is_chance = mk in AT_CHANCE_MODELS
            ax.bar(
                x + i * width, vals, width,
                label=(MODEL_LABELS[mk] + ' (at chance)') if is_chance else MODEL_LABELS[mk],
                color='#999999' if is_chance else MODEL_COLORS[mk],
                alpha=0.55 if is_chance else 0.85,
                hatch='///' if is_chance else None,
                edgecolor='white',
            )
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in PROMPTING_STRATEGIES])
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    _draw(
        axes[0],
        lambda mk, s: strategy_results[(mk, s)]['accuracy'],
        'Overall Accuracy by Prompting Strategy',
    )
    _draw(
        axes[1],
        lambda mk, s: strategy_results[(mk, s)]['by_level']['L2'],
        'L2 (Intervention) Accuracy by Prompting Strategy',
    )

    axes[0].text(
        0.01, -0.22,
        'GPT-2 Small uses raw-generation log-prob; others use PMI-debiased. '
        'GPT-2 bars are greyed/hatched — debiased 95% CIs include chance (25%).',
        transform=axes[0].transAxes, fontsize=7, color='#555555',
    )

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
