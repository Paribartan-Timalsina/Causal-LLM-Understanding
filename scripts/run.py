"""Run the full evaluation: build benchmark, load models, score, save results.

Examples:
    python -m scripts.run
    python -m scripts.run --models qwen_1_5b llama_3_3b
    python -m scripts.run --skip-strategies
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from causal_llm import (
    MODEL_CONFIGS,
    MODEL_LABELS,
    OUTPUT_DIR,
    PROMPTING_STRATEGIES,
    SEED,
    STRATEGY_LABELS,
    benchmark_summary,
    build_benchmark,
    evaluate_model,
    load_models,
    plot_accuracy_heatmap,
    plot_benchmark_stats,
    plot_causal_graphs,
    plot_strategy_comparison,
    save_results,
    setup_hf_auth,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    p.add_argument('--models', nargs='*', default=None,
                   choices=list(MODEL_CONFIGS.keys()),
                   help='subset of models to run (default: all)')
    p.add_argument('--strategy-bucket', type=int, default=10,
                   help='questions per (graph,level) bucket for strategy comparison '
                        '(10 = 120 total, 20 = 240 = full benchmark)')
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--skip-strategies', action='store_true',
                   help='only run zero-shot, skip the strategy sweep')
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed
        set_seed(seed)
    except Exception:
        pass


def stratified_sample(benchmark, per_bucket):
    by_cell = defaultdict(list)
    for q in benchmark:
        by_cell[(q['graph'], q['level'])].append(q)
    out = []
    for key in sorted(by_cell.keys()):
        out.extend(by_cell[key][:per_bucket])
    return out


def run_zero_shot(active_models, models, tokenizers, benchmark, device):
    by_level = {lv: [q for q in benchmark if q['level'] == lv] for lv in ['L1', 'L2', 'L3']}
    all_results = {}
    accuracy_matrix = {}

    for mk in active_models:
        accuracy_matrix[mk] = {}
        print(f"\n{'=' * 60}\nEvaluating {MODEL_LABELS[mk]}\n{'=' * 60}")
        for lv in ['L1', 'L2', 'L3']:
            results, acc = evaluate_model(mk, by_level[lv], 'zero_shot',
                                          models, tokenizers, device)
            all_results[(mk, lv)] = (results, acc)
            accuracy_matrix[mk][lv] = acc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results, accuracy_matrix


def print_zero_shot_summary(active_models, accuracy_matrix):
    print(f"\n{'=' * 60}\nZERO-SHOT ACCURACY SUMMARY\n{'=' * 60}")
    print(f"{'Model':25s} | {'L1 (Assoc)':>12s} | {'L2 (Interv)':>12s} | {'L3 (Counter)':>12s}")
    print('-' * 70)
    for mk in active_models:
        a = accuracy_matrix[mk]
        print(f"{MODEL_LABELS[mk]:25s} | {a['L1']:>11.1%} | {a['L2']:>11.1%} | {a['L3']:>11.1%}")
    print('-' * 70)
    print(f"{'Random baseline':25s} | {'25.0%':>12s} | {'25.0%':>12s} | {'25.0%':>12s}")


def compute_detailed_acc(active_models, all_results):
    acc = defaultdict(lambda: defaultdict(dict))
    for (mk, lv), (results, _) in all_results.items():
        for gt in {r['graph'] for r in results}:
            graph_results = [r for r in results if r['graph'] == gt]
            if graph_results:
                acc[mk][lv][gt] = float(np.mean([r['correct'] for r in graph_results]))
    return {mk: dict(acc[mk]) for mk in active_models}


def run_strategies(active_models, models, tokenizers, sample, device):
    out = {}
    for mk in active_models:
        print(f"\n{'=' * 60}\n{MODEL_LABELS[mk]} -- Prompting Strategy Comparison\n{'=' * 60}")
        for s in PROMPTING_STRATEGIES:
            results, acc = evaluate_model(mk, sample, s, models, tokenizers, device)
            by_level = {
                lv: float(np.mean([r['correct'] for r in results if r['level'] == lv]))
                if any(r['level'] == lv for r in results) else 0.0
                for lv in ['L1', 'L2', 'L3']
            }
            out[(mk, s)] = {'accuracy': acc, 'results': results, 'by_level': by_level}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def print_strategy_summary(active_models, strategy_results):
    header = f"{'Strategy':16s}"
    for mk in active_models:
        header += f' | {MODEL_LABELS[mk]:>25s}'
    print('\n' + header)
    print('-' * 100)
    for s in PROMPTING_STRATEGIES:
        row = f'{STRATEGY_LABELS[s]:16s}'
        for mk in active_models:
            row += f' | {strategy_results[(mk, s)]["accuracy"]:>24.1%}'
        print(row)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    setup_hf_auth()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    benchmark = build_benchmark(seed=args.seed)
    summary = benchmark_summary(benchmark)
    print(f"Benchmark: {summary['total']} questions, "
          f"{summary['duplicates']} duplicate fingerprints, "
          f"answers {summary['answers']}")

    plot_causal_graphs(args.output_dir / 'causal_graphs.png')
    plot_benchmark_stats(benchmark, args.output_dir / 'benchmark_stats.png')

    configs = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models} \
        if args.models else MODEL_CONFIGS
    loaded = load_models(device, configs=configs)
    if not loaded.models:
        print('No models loaded; aborting.', file=sys.stderr)
        return 1

    all_results, accuracy_matrix = run_zero_shot(
        loaded.active, loaded.models, loaded.tokenizers, benchmark, device,
    )
    print_zero_shot_summary(loaded.active, accuracy_matrix)
    detailed_acc = compute_detailed_acc(loaded.active, all_results)
    plot_accuracy_heatmap(detailed_acc, loaded.active,
                          args.output_dir / 'accuracy_by_graph_level.png')

    if args.skip_strategies:
        # Empty placeholders so save_results can still write the manifest etc.
        strategy_results = {}
        for mk in loaded.active:
            for s in PROMPTING_STRATEGIES:
                strategy_results[(mk, s)] = {
                    'accuracy': 0.0,
                    'results': [],
                    'by_level': {'L1': 0.0, 'L2': 0.0, 'L3': 0.0},
                }
    else:
        sample = stratified_sample(benchmark, args.strategy_bucket)
        print(f'\nStrategy sample: {len(sample)} questions '
              f'({args.strategy_bucket} per (graph, level) bucket).')
        strategy_results = run_strategies(
            loaded.active, loaded.models, loaded.tokenizers, sample, device,
        )
        print_strategy_summary(loaded.active, strategy_results)
        plot_strategy_comparison(strategy_results, loaded.active,
                                 args.output_dir / 'prompting_strategy_comparison.png')

    save_results(
        output_dir=args.output_dir,
        benchmark=benchmark,
        accuracy_matrix=accuracy_matrix,
        detailed_acc=detailed_acc,
        all_results=all_results,
        strategy_results=strategy_results,
        active_models=loaded.active,
        skipped_models=loaded.skipped,
        seed=args.seed,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
