# Can LLMs Reason Causally? Probing Causal Inference Across Pearl's Hierarchy

Systematic evaluation of causal reasoning in five LLMs (124M to 3.2B params) across the three levels of Pearl's causal hierarchy and four canonical graph structures.

## What it does

For each (model × graph × level × prompting strategy) combination, the pipeline:

1. Generates a 240-question multiple-choice benchmark with PMI-debiased ground-truth answers
2. Loads each model (4-bit NF4 for instruct models on GPU)
3. Scores each answer letter via forced-completion log-probability with content-free PMI calibration
4. Reports accuracy by graph type and reasoning level + a four-way prompting-strategy comparison

## Models evaluated

| Model | Params | Type | Quant |
|---|---|---|---|
| GPT-2 Small | 124M | Base | FP32 |
| GPT-2 Large | 774M | Base | FP32 |
| Qwen2.5-1.5B-Instruct | 1.5B | Instruct | 4-bit NF4 |
| Llama-3.2-3B-Instruct | 3.2B | Instruct | 4-bit NF4 |
| Gemma-2-2B-it | 2.6B | Instruct | 4-bit NF4 |

Llama and Gemma are license-gated; set `HF_TOKEN` (env var, Kaggle Secret, or Colab userdata).

## Benchmark

Four canonical causal structures × three reasoning levels × 20 questions = 240 prompts. Choice labels are permuted per question to defeat letter-frequency priors.

| Structure | Graph | Key property |
|---|---|---|
| Chain | X → Y → Z | Mediation |
| Fork | X ← C → Z | Common cause / confounding |
| Collider | X → M ← Z | Explaining away |
| Diamond | X → {Y,Z} → W | Multiple pathways |

| Level | Name | Question type |
|---|---|---|
| L1 | Association | P(Y \| X) - observation |
| L2 | Intervention | P(Y \| do(X)) - manipulation |
| L3 | Counterfactual | P(Y_x \| X′,Y′) - would-have-been |

## Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Python ≥3.9, PyTorch ≥2.0. CUDA GPU strongly recommended (~16 GB VRAM for the full set; T4 fits all five with NF4).

## Run

```bash
# Full evaluation (all 5 models, ~75 min on T4)
python -m scripts.run

# Subset of models
python -m scripts.run --models qwen_1_5b llama_3_3b

# Custom output directory
python -m scripts.run --output-dir runs/v1

# Skip strategy comparison (zero-shot only)
python -m scripts.run --skip-strategies

# Full benchmark for strategy comparison (240 questions instead of 120)
python -m scripts.run --strategy-bucket 20
```

## Outputs

Written to `output_llm/` (or `--output-dir`):

| File | Contents |
|---|---|
| `results_summary.json` | Machine-readable: zero-shot accuracy, per-graph breakdown, per-strategy accuracy |
| `zero_shot_accuracy.csv` | Model × level |
| `strategy_accuracy.csv` | Model × strategy × level |
| `detailed_accuracy.csv` | Model × level × graph |
| `raw_predictions.jsonl` | Per-question rows for both zero-shot and strategy runs |
| `accuracy_by_graph_level.png` | Heatmap |
| `prompting_strategy_comparison.png` | Bar charts |
| `causal_graphs.png`, `benchmark_stats.png` | Reference figures |
| `run_manifest.json` | Versions, GPU, seed, timestamp |

## Layout

```
causal_llm/
  config.py         # Model configs, paths, seed, prompting strategies
  graphs.py         # CAUSAL_GRAPHS dict + plot_causal_graphs
  scenarios.py      # Realistic story templates per graph type
  benchmark.py      # make_question, build_benchmark
  models.py         # HF auth + load_models with 4-bit NF4
  prompts.py        # Four prompt formatters + content-free prompts
  scoring.py        # Forced-completion log-prob + calibration priors
  evaluator.py      # Per-(model,strategy) eval loop with PMI / generation routing
  visualization.py  # Heatmap and bar-chart plots
  reporting.py      # JSON / CSV / JSONL serialization
scripts/run.py      # CLI entry point
tests/              # Smoke tests for benchmark + prompts (no GPU needed)
```

## Testing

```bash
python -m pytest tests/ -v
```

The included tests are CPU-only (benchmark generation, fingerprinting, prompt formatting). Model evaluation is exercised via `python -m scripts.run`.

## Notes on scoring

- **PMI calibration** subtracts the model's letter prior (measured against an N/A content-free prompt) from each raw log-prob to remove token-frequency bias.
- **Generation fallback** kicks in when calibration priors are NaN/collapsed, or when a base LM has a prior range so wide that PMI's linear correction can't undo the nonlinear softmax bias (typical for GPT-2 Small/Large).
- **GPT-2 results sit at chance** (debiased CIs include 25%) - they're a calibration baseline, not a reasoning result. Cross-model comparisons against them are not apples-to-apples.
