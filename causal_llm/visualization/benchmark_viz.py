"""Visualization for benchmark statistics and causal graph structures."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from ..config import LEVEL_COLORS, OUTPUT_DIR
from ..benchmark.graphs import CAUSAL_GRAPHS, LAYOUT_OVERRIDES


def plot_causal_graphs(output_dir=None):
    """Visualize the five canonical causal structures.

    Args:
        output_dir: Directory to save figure. Defaults to OUTPUT_DIR.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for idx, (gname, ginfo) in enumerate(CAUSAL_GRAPHS.items()):
        ax = axes[idx]
        G = nx.DiGraph()
        G.add_edges_from(ginfo["edges"])
        pos = LAYOUT_OVERRIDES[gname]

        node_colors = []
        for n in G.nodes():
            if n == "U":
                node_colors.append("#FFB3BA")
            else:
                node_colors.append("#BAE1FF")

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=800, edgecolors="black", linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=13, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#333333",
                               arrows=True, arrowsize=20, width=2,
                               connectionstyle="arc3,rad=0.1")
        ax.set_title(f'{ginfo["description"]}', fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Five Canonical Causal Structures", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "causal_graphs.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_benchmark_stats(benchmark, output_dir=None):
    """Plot benchmark distribution statistics.

    Args:
        benchmark: list of question dicts
        output_dir: Directory to save figure.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    df = pd.DataFrame(benchmark)
    ct = pd.crosstab(df["graph"], df["level"])
    ct = ct[["L1", "L2", "L3"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ct.plot(kind="bar", ax=axes[0],
            color=[LEVEL_COLORS["L1"], LEVEL_COLORS["L2"], LEVEL_COLORS["L3"]])
    axes[0].set_title("Questions by Graph Type & Level", fontweight="bold")
    axes[0].set_xlabel("Causal Graph Type")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="Level")

    level_counts = df["level"].value_counts().sort_index()
    axes[1].pie(level_counts, labels=level_counts.index,
                colors=[LEVEL_COLORS[l] for l in level_counts.index],
                autopct="%1.0f%%", startangle=90, textprops={"fontsize": 12})
    axes[1].set_title("Distribution by Reasoning Level", fontweight="bold")

    ans_counts = df["answer"].value_counts().sort_index()
    axes[2].bar(ans_counts.index, ans_counts.values, color="#7FB3D8")
    axes[2].set_title("Answer Key Distribution", fontweight="bold")
    axes[2].set_xlabel("Correct Answer")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
