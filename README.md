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

## Summary of findings

Numbers below come from a single full run on a Kaggle T4 GPU.
 
**Zero-shot accuracy on the full 240-question benchmark:**

| Model | L1 (Assoc) | L2 (Interv) | L3 (Counter) |
|---|---:|---:|---:|
| GPT-2 Small (124M)     | 21% | 29% | 24% |
| GPT-2 Large (774M)     | 21% | 20% | 25% |
| Qwen2.5-1.5B-Instruct  | **72%** | 58% | 64% |
| Llama-3.2-3B-Instruct  | 70% | 51% | 65% |
| Gemma-2-2B-it          | 60% | 35% | 50% |
| *Random baseline*      | *25%* | *25%* | *25%* |

**Prompting strategy comparison (overall, 120-question subset, best in bold):**

| Model | Zero-Shot | Few-Shot | CoT | Causal Chain |
|---|---:|---:|---:|---:|
| GPT-2 Small (at chance)    | 22% | **25%** | 20% | 23% |
| GPT-2 Large (at chance)    | 23% | **27%** | 19% | 18% |
| Qwen2.5-1.5B-Instruct      | 66% | **70%** | 67% | 43% |
| Llama-3.2-3B-Instruct      | 64% | **66%** | 61% | 58% |
| Gemma-2-2B-it              | 53% | **69%** | 56% | 31% |
| *Random baseline*          | *25%* | *25%* | *25%* | *25%* |

1. **Both GPT-2 models are at chance.** Their debiased 95% CIs include 25%. They predict a single letter on most questions (GPT-2 Small picks `A` on 67% of prompts; GPT-2 Large picks `C` or `D` on 93%). Treat them as a calibration baseline, not a reasoning result.

2. **L2 (Intervention) is harder than L3 (Counterfactual) for every instruct model.** Pearl's hierarchy predicts L1 < L2 < L3 in difficulty; we see the opposite at L2/L3. The drop from L1 to L2 is 15-25 percentage points and is partially recovered at L3. Plausible reading: the L3 questions provide richer scenario-grounded narratives, while bare `do(X)` framing pulls the model into associational shortcuts.

3. **Collider ("explaining away") is universally the hardest graph.** Average accuracy across L1/L2/L3:

   | Model | chain | fork | **collider** | diamond |
   |---|---:|---:|---:|---:|
   | Qwen   | 82% | 85% | **35%** | 57% |
   | Llama  | 92% | 65% | **13%** | 78% |
   | Gemma  | 63% | 48% | **23%** | 58% |

   Llama scores 0% on L3 + collider. Models recognize directed-influence chains but fail at conditioning-on-effect reasoning, which is the textbook hard case.

4. **Few-shot helps everywhere; "Causal Chain" prompting often hurts.** Few-shot is the only strategy that improves every instruct model (Gemma jumps +17pp). Chain-of-thought is roughly neutral. The Causal Chain template collapses Qwen by -22pp and Gemma by -22pp - the long structured prompt skews the letter prior so heavily that PMI can't undo it ("prior collapse").

5. **Llama vs Qwen flip on graph type.** Llama (3.2B) is bigger but lower overall zero-shot than Qwen (1.5B). Llama dominates chain (92%) and diamond (78%); Qwen dominates fork (85%) and is roughly 2x better on collider (35% vs 13%). Qwen is the more balanced generalist on this benchmark.

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

## Notes on scoring

- **PMI calibration** subtracts the model's letter prior (measured against an N/A content-free prompt) from each raw log-prob to remove token-frequency bias.
- **Generation fallback** kicks in when calibration priors are NaN/collapsed, or when a base LM has a prior range so wide that PMI's linear correction can't undo the nonlinear softmax bias (typical for GPT-2 Small/Large).
- **GPT-2 results sit at chance** (debiased CIs include 25%) - they're a calibration baseline, not a reasoning result. Cross-model comparisons against them are not apples-to-apples.
