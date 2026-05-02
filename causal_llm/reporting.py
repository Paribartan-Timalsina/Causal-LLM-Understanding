"""Write JSON / CSV / JSONL output files for a completed run."""

import csv
import datetime as dt
import json

import numpy as np
import torch

from .config import MODEL_CONFIGS, MODEL_LABELS, PROMPTING_STRATEGIES
from .graphs import CAUSAL_GRAPHS


def save_results(output_dir, benchmark, accuracy_matrix, detailed_acc,
                 all_results, strategy_results, active_models, skipped_models, seed,
                 bootstrap_summary=None, gen_cot_results=None):
    """Write summary JSON, per-evaluation CSVs, and per-question JSONLs.

    `bootstrap_summary` and `gen_cot_results` are optional. When provided,
    `permuted_accuracy.csv` / `gen_cot_accuracy.csv` / `gen_cot_chains.jsonl`
    are written and the corresponding blocks are added to `results_summary.json`.
    """
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

    if bootstrap_summary:
        summary['permuted_accuracy'] = {
            f'{mk}__{lv}': {
                'mean': bootstrap_summary[(mk, lv)]['mean'],
                'lo': bootstrap_summary[(mk, lv)]['lo'],
                'hi': bootstrap_summary[(mk, lv)]['hi'],
                'single_perm': bootstrap_summary[(mk, lv)]['single_perm'],
                'n': bootstrap_summary[(mk, lv)]['n'],
            }
            for mk in active_models for lv in ['L1', 'L2', 'L3']
            if (mk, lv) in bootstrap_summary
        }

    if gen_cot_results:
        summary['gen_cot_accuracy'] = {
            mk: {
                'accuracy': r['accuracy'],
                'ci': r['ci'],
                'by_level': r['by_level'],
                'extract_rate': r['extract_rate'],
            }
            for mk, r in gen_cot_results.items()
        }

    _dump_json(output_dir / 'results_summary.json', summary)

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
                    w.writerow([MODEL_LABELS[mk], lv, gt,
                                detailed_acc[mk].get(lv, {}).get(gt, 0)])
    print('Saved', output_dir / 'detailed_accuracy.csv')

    if bootstrap_summary:
        with open(output_dir / 'permuted_accuracy.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'level', 'single_perm', 'permuted_mean',
                        'ci_lo', 'ci_hi', 'n'])
            for mk in active_models:
                for lv in ['L1', 'L2', 'L3']:
                    s = bootstrap_summary.get((mk, lv))
                    if s is None:
                        continue
                    w.writerow([MODEL_LABELS[mk], lv, s['single_perm'], s['mean'],
                                s['lo'], s['hi'], s['n']])
        print('Saved', output_dir / 'permuted_accuracy.csv')

    if gen_cot_results:
        with open(output_dir / 'gen_cot_accuracy.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'accuracy', 'ci_lo', 'ci_hi',
                        'L1', 'L2', 'L3', 'extract_rate'])
            for mk, r in gen_cot_results.items():
                ci = r['ci']
                bl = r['by_level']
                w.writerow([MODEL_LABELS[mk], r['accuracy'], ci['lo'], ci['hi'],
                            bl['L1'], bl['L2'], bl['L3'], r['extract_rate']])
        print('Saved', output_dir / 'gen_cot_accuracy.csv')

    rows = []
    for (mk, _lv), (results, _acc) in all_results.items():
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
    if gen_cot_results:
        for mk, gr in gen_cot_results.items():
            for r in gr['raw']:
                rows.append({
                    'source': 'gen_cot',
                    'model': mk,
                    'strategy': 'generation_cot',
                    'qid': r['qid'],
                    'graph': r['graph'],
                    'level': r['level'],
                    'prediction': r['prediction'],
                    'correct_letter': r['gold'],
                    'correct': r['correct'],
                })
    with open(output_dir / 'raw_predictions.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + '\n')
    print(f'Saved {len(rows)} rows to', output_dir / 'raw_predictions.jsonl')

    if gen_cot_results:
        with open(output_dir / 'gen_cot_chains.jsonl', 'w') as f:
            for mk, gr in gen_cot_results.items():
                for r in gr['raw']:
                    f.write(json.dumps({**r, 'model': mk}, default=str) + '\n')
        print('Saved', output_dir / 'gen_cot_chains.jsonl')

    manifest = {
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
    _dump_json(output_dir / 'run_manifest.json', manifest)


def _dump_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print('Saved', path)
