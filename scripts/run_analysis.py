"""Run internal representation analysis (linear probing and attention)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_llm.config import set_seed, get_device, SEED, OUTPUT_DIR, PROBE_NUM_SAMPLES
from causal_llm.benchmark import generate_benchmark
from causal_llm.models import load_model
from causal_llm.evaluation.prompts import format_zero_shot
from causal_llm.analysis import extract_hidden_states, run_probing, extract_attention


def main():
    parser = argparse.ArgumentParser(description="Run probing and attention analysis.")
    parser.add_argument("--models", nargs="+", default=["gpt2_small", "gpt2_large"],
                        help="Model keys for probing analysis.")
    parser.add_argument("--num-samples", type=int, default=PROBE_NUM_SAMPLES,
                        help="Number of benchmark samples for probing.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    benchmark = generate_benchmark(seed=args.seed)
    probe_questions = benchmark[:args.num_samples]

    graph_type_map = {"chain": 0, "fork": 1, "collider": 2, "diamond": 3, "instrument": 4}
    level_map = {"L1": 0, "L2": 1, "L3": 2}

    prompts = [format_zero_shot(q) for q in probe_questions]
    graph_labels = np.array([graph_type_map[q["graph"]] for q in probe_questions])
    level_labels = np.array([level_map[q["level"]] for q in probe_questions])

    all_probing_results = {}

    for model_key in args.models:
        print(f"\n--- Probing {model_key} ---")
        model, tokenizer = load_model(model_key, device=device)

        print("  Extracting hidden states...")
        hidden = extract_hidden_states(model, tokenizer, prompts, device)

        print("  Running graph-type probing...")
        graph_probe = run_probing(hidden, graph_labels, "graph_type")

        print("  Running level probing...")
        level_probe = run_probing(hidden, level_labels, "reasoning_level")

        all_probing_results[model_key] = {
            "graph_type": graph_probe,
            "reasoning_level": level_probe,
        }

        print("  Extracting sample attention patterns...")
        sample_prompt = prompts[0]
        attn_weights, tokens = extract_attention(model, tokenizer, sample_prompt, device)
        print(f"    Attention shape: {attn_weights.shape}")

    results_path = output_dir / "probing_results.json"
    serializable = {}
    for mk, data in all_probing_results.items():
        serializable[mk] = {}
        for task, probe_res in data.items():
            serializable[mk][task] = {
                k: {str(layer): float(v) for layer, v in vals.items()}
                for k, vals in probe_res.items()
            }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nProbing results saved to {results_path}")


if __name__ == "__main__":
    main()
