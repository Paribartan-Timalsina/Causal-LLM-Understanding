#!/usr/bin/env python3
"""Generate comprehensive visualization report from saved results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_llm.config import set_seed, SEED, OUTPUT_DIR
from causal_llm.benchmark import generate_benchmark
from causal_llm.visualization import (
    plot_benchmark_stats,
    plot_causal_graphs,
)


def main():
    parser = argparse.ArgumentParser(description="Generate visualization report.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory with results.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Generating causal graph visualizations...")
    plot_causal_graphs(output_dir=output_dir)

    print("Generating benchmark statistics...")
    benchmark = generate_benchmark(seed=args.seed)
    plot_benchmark_stats(benchmark, output_dir=output_dir)

    results_path = output_dir / "results_summary.json"
    if results_path.exists():
        print("Loading evaluation results for dashboard...")
        with open(results_path) as f:
            summary = json.load(f)
        print(f"  Found results for: {list(summary.get('results', {}).keys())}")
        print("  (Full dashboard requires running evaluation first.)")
    else:
        print(f"  No results found at {results_path}.")
        print("  Run `python scripts/run_evaluation.py` first to generate results.")

    print(f"\nAll output files saved to {output_dir}/")
    for p in sorted(output_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
