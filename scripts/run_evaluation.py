"""Main CLI entry point: generate benchmark, load models, evaluate, save results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_llm.config import (
    set_seed, get_device, SEED, OUTPUT_DIR, PROMPTING_STRATEGIES,
    MODEL_CONFIGS, MODEL_LABELS,
)
from causal_llm.benchmark import generate_benchmark
from causal_llm.models import load_model
from causal_llm.evaluation import evaluate_model, run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM causal reasoning.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        help="Model keys to evaluate.")
    parser.add_argument("--strategies", nargs="+", default=["zero_shot"],
                        help="Prompting strategies.")
    parser.add_argument("--num-questions", type=int, default=20,
                        help="Questions per (graph, level) cell.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.device:
        import torch
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    print("\n--- Generating benchmark ---")
    benchmark = generate_benchmark(num_per_cell=args.num_questions, seed=args.seed)
    print(f"Generated {len(benchmark)} questions")

    loaded_models = {}
    loaded_tokenizers = {}
    for model_key in args.models:
        try:
            model, tokenizer = load_model(model_key, device=device)
            loaded_models[model_key] = model
            loaded_tokenizers[model_key] = tokenizer
        except Exception as e:
            print(f"  Failed to load {model_key}: {e}")

    print("\n--- Running evaluation ---")
    all_results = run_full_evaluation(
        loaded_models, loaded_tokenizers, benchmark,
        strategies=args.strategies, device=device, verbose=True,
    )

    results_summary = {
        "benchmark_size": len(benchmark),
        "models": list(loaded_models.keys()),
        "strategies": args.strategies,
        "results": {},
    }

    for (mk, strategy), data in all_results.items():
        key = f"{mk}__{strategy}"
        results_summary["results"][key] = {
            "accuracy": data["accuracy"],
            "num_questions": len(data["results"]),
            "by_level": {},
            "by_graph": {},
        }
        for level in ["L1", "L2", "L3"]:
            lvl_res = [r for r in data["results"] if r["level"] == level]
            if lvl_res:
                results_summary["results"][key]["by_level"][level] = {
                    "accuracy": sum(r["correct"] for r in lvl_res) / len(lvl_res),
                    "count": len(lvl_res),
                }
        for graph in ["chain", "fork", "collider", "diamond", "instrument"]:
            g_res = [r for r in data["results"] if r["graph"] == graph]
            if g_res:
                results_summary["results"][key]["by_graph"][graph] = {
                    "accuracy": sum(r["correct"] for r in g_res) / len(g_res),
                    "count": len(g_res),
                }

    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
