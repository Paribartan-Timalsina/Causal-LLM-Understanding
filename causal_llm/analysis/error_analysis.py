"""Error analysis: confusion matrices and failure pattern identification."""

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrices(results_by_model):
    """Compute normalized confusion matrices for each model.

    Args:
        results_by_model: dict mapping model_key -> list of result dicts

    Returns:
        dict: {model_key: {'matrix': np.array, 'labels': list}}
    """
    labels = ["A", "B", "C", "D"]
    matrices = {}

    for model_key, results in results_by_model.items():
        y_true = [r["answer"] for r in results]
        y_pred = [r["prediction"] for r in results]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm / row_sums

        matrices[model_key] = {
            "matrix": cm_normalized,
            "labels": labels,
            "raw_matrix": cm,
        }

    return matrices


def identify_failure_patterns(results):
    """Identify systematic failure patterns across graph types and levels.

    Args:
        results: list of result dicts

    Returns:
        dict: {
            'hardest_cells': list of (graph, level, accuracy) sorted by difficulty,
            'prediction_bias': Counter of prediction frequencies,
            'by_graph': dict of graph -> accuracy,
            'by_level': dict of level -> accuracy,
        }
    """
    cell_correct = defaultdict(int)
    cell_total = defaultdict(int)
    graph_correct = defaultdict(int)
    graph_total = defaultdict(int)
    level_correct = defaultdict(int)
    level_total = defaultdict(int)
    predictions = []

    for r in results:
        key = (r["graph"], r["level"])
        cell_total[key] += 1
        cell_correct[key] += int(r["correct"])
        graph_total[r["graph"]] += 1
        graph_correct[r["graph"]] += int(r["correct"])
        level_total[r["level"]] += 1
        level_correct[r["level"]] += int(r["correct"])
        predictions.append(r["prediction"])

    hardest = []
    for key in cell_total:
        acc = cell_correct[key] / cell_total[key] if cell_total[key] > 0 else 0
        hardest.append((key[0], key[1], acc))
    hardest.sort(key=lambda x: x[2])

    by_graph = {g: graph_correct[g] / graph_total[g] for g in graph_total}
    by_level = {l: level_correct[l] / level_total[l] for l in level_total}

    return {
        "hardest_cells": hardest,
        "prediction_bias": Counter(predictions),
        "by_graph": by_graph,
        "by_level": by_level,
    }
