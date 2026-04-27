"""Result serialization: JSON summaries, CSV tables, JSONL raw predictions, manifest."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from .config import MODEL_CONFIGS, MODEL_LABELS, PROMPTING_STRATEGIES
from .graphs import CAUSAL_GRAPHS


def save_results(
    output_dir: Path,
    benchmark: list[dict],
    accuracy_matrix: dict,
    detailed_acc: dict,
    all_results: dict,
    strategy_results: dict,
    active_models: dict,
    skipped_models: list,
    seed: int,
) -> None:
    """Write results_summary.json + 3 CSVs + raw_predictions.jsonl + run_manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'benchmark_size': len(benchmark),
        'models': list(active_models.keys()),
        'skipped_models': skipped_models,
        'seed': seed,
        'zero_shot_accuracy': {mk: accuracy_matrix[mk] for mk in active_models},
        'detailed_accuracy': {
            mk: {
                lv: {gt: detailed_acc[mk].get(lv, {}).get(gt, 0) for gt in CAUSAL_GRAPHS}
                for lv in ['L1', 'L2', 'L3']
            }
            for mk in active_models
        },
        'strategy_accuracy': {
            f'{mk}__{s}': strategy_results[(mk, s)]['accuracy']
            for mk in active_models for s in PROMPTING_STRATEGIES
        },
        'strategy_by_level': {
            f'{mk}__{s}': strategy_results[(mk, s)]['by_level']
            for mk in active_models for s in PROMPTING_STRATEGIES
        },
    }
    _write_json(output_dir / 'results_summary.json', summary)

    with open(output_dir / 'zero_shot_accuracy.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'L1', 'L2', 'L3'])
        for mk in active_models:
            a = accuracy_matrix[mk]
            w.writerow([MODEL_LABELS[mk], a['L1'], a['L2'], a['L3']])
    print('Saved', output_dir / 'zero_shot_accuracy.csv')

    with open(output_dir / 'strategy_accuracy.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'strategy', 'overall', 'L1', 'L2', 'L3'])
        for mk in active_models:
            for s in PROMPTING_STRATEGIES:
                sr = strategy_results[(mk, s)]
                bl = sr['by_level']
                w.writerow([MODEL_LABELS[mk], s, sr['accuracy'], bl['L1'], bl['L2'], bl['L3']])
    print('Saved', output_dir / 'strategy_accuracy.csv')

    with open(output_dir / 'detailed_accuracy.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'level', 'graph', 'accuracy'])
        for mk in active_models:
            for lv in ['L1', 'L2', 'L3']:
                for gt in CAUSAL_GRAPHS:
                    acc = detailed_acc[mk].get(lv, {}).get(gt, 0)
                    w.writerow([MODEL_LABELS[mk], lv, gt, acc])
    print('Saved', output_dir / 'detailed_accuracy.csv')

    rows = []
    for (mk, lv), (results, _acc) in all_results.items():
        for r in results:
            rows.append({
                'source': 'zero_shot_full',
                'model': mk,
                'strategy': 'zero_shot',
                'qid': r['id'],
                'graph': r['graph'],
                'level': r['level'],
                'prediction': r['prediction'],
                'correct_letter': r['answer'],
                'correct': r['correct'],
            })
    for (mk, s), sr in strategy_results.items():
        for r in sr['results']:
            rows.append({
                'source': 'strategy_subset',
                'model': mk,
                'strategy': s,
                'qid': r['id'],
                'graph': r['graph'],
                'level': r['level'],
                'prediction': r['prediction'],
                'correct_letter': r['answer'],
                'correct': r['correct'],
            })
    with open(output_dir / 'raw_predictions.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + '\n')
    print(f'Saved {len(rows)} rows to', output_dir / 'raw_predictions.jsonl')

    manifest: dict = {
        'seed': seed,
        'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
        'models': {mk: MODEL_CONFIGS[mk]['name'] for mk in active_models},
    }
    try:
        import transformers
        manifest['versions'] = {
            'transformers': transformers.__version__,
            'torch': torch.__version__,
            'numpy': np.__version__,
        }
    except Exception as e:
        manifest['versions_error'] = str(e)
    if torch.cuda.is_available():
        manifest['gpu'] = torch.cuda.get_device_name(0)
        manifest['cuda'] = getattr(torch.version, 'cuda', None)  # type: ignore[attr-defined]
    else:
        manifest['gpu'] = 'cpu'
    _write_json(output_dir / 'run_manifest.json', manifest)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print('Saved', path)
