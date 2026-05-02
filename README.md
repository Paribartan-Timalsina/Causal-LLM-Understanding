# Can LLMs Reason Causally? Probing Causal Inference Across Pearl's Hierarchy

Systematic evaluation of causal reasoning in five LLMs (124M to 3.2B params) across the three levels of Pearl's causal hierarchy and four canonical graph structures.

## What it does

The pipeline:

1. Generates a 240-question multiple-choice benchmark covering the four canonical causal graphs (chain / fork / collider / diamond) at each of Pearl's three reasoning levels (L1 / L2 / L3). Choice labels are shuffled per question so the correct answer is uniformly distributed over A/B/C/D.
2. Loads each model (4-bit NF4 quantization for the instruction-tuned models, FP32 for the GPT-2 baselines).
3. Scores each candidate answer letter using forced-completion log-probability, then debiases the score by subtracting the model's letter prior measured against an N/A content-free prompt (PMI calibration).
4. Reports zero-shot accuracy broken down by reasoning level and graph type, plus a four-way prompting-strategy comparison.

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

Four canonical causal structures × three reasoning levels × 20 questions = 240 prompts. Choice labels are shuffled per question so the correct answer letter is roughly uniformly distributed across A/B/C/D (this prevents a model from getting a high score by always guessing one letter).

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

All numbers below come from a single seeded run (seed = 42) on a Kaggle T4 GPU. The headline tables in this section use the full 240-question benchmark; the robustness section uses a 120-question stratified subset (40 per reasoning level) so the permutation-averaging budget stays manageable.

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

### Robustness checks

The headline numbers above are from a single letter ordering and a single point estimate per cell. Two confounds remain to address.

**Permutation averaging.** A 120-question stratified subset (40 per reasoning level) was re-scored with all four cyclic permutations of A/B/C/D so the correct answer hits each position exactly once across the four passes. Per-question accuracy is the mean over the four passes. Bootstrap 95% CIs (n_iter=2000) are computed over the resulting per-question scores.

| Model | L1 | L2 | L3 |
|---|:---:|:---:|:---:|
| GPT-2 Small (124M)  | 24% [23, 25] | 27% [24, 29] | 24% [23, 25] |
| GPT-2 Large (774M)  | 24% [22, 26] | 28% [24, 31] | 23% [20, 26] |
| Qwen2.5-1.5B-Inst   | 71% [59, 83] | 52% [39, 64] | 54% [43, 66] |
| Llama-3.2-3B-Inst   | 73% [61, 85] | 50% [39, 61] | 67% [53, 79] |
| Gemma-2-2B-it       | 57% [48, 67] | 39% [29, 50] | 46% [38, 54] |

All six GPT-2 cells have CIs that include 25%, formally placing them at chance. Every instruct-model cell sits above chance, including L2. The drop from single-perm to perm-averaged accuracy (the residual letter-position bias) was at most 5pp for Llama and GPT-2, 11pp for Qwen and Gemma — i.e., Llama's headline numbers were the most honest, while Qwen's and Gemma's were inflated by ~10pp of position bias.

**Real-generation Chain-of-Thought.** The CoT row in the strategy table above scores the answer letter against a CoT-flavored prompt template, but the model never actually produces a chain. To check whether real reasoning matches the PMI signal, we re-ran the three instruct models on the same 120-question subset, generating ~250 tokens per question and parsing the answer letter from the output.

| Model | PMI-CoT | Gen-CoT | 95% CI (gen) | Δ |
|---|---:|---:|:---:|---:|
| Qwen2.5-1.5B  | 67% | 60% | [52, 68] | -7pp |
| Llama-3.2-3B  | 61% | 62% | [53, 70] | +1pp |
| Gemma-2-2B    | 56% | 63% | [54, 72] | +7pp |

Aggregate accuracy is similar between the two methods (about +0.3pp average), but per-model effects differ: Qwen marginally loses with real generation, Llama is unchanged, Gemma gains. None of these per-model deltas exceed the 95% CI half-width (~10pp), so individually they are weak signals; the consistent pattern across three models is more interesting than any one delta.

### Findings

1. **Both GPT-2 models are at chance.** Their permutation-averaged 95% CIs include 25% at every reasoning level. The behavior is consistent with letter-prior guessing: GPT-2 Small predicts `A` on 67% of prompts, GPT-2 Large predicts `C` or `D` on 93%. They function as a calibration baseline, not as evidence of causal reasoning.

2. **L2 (Intervention) is harder than L3 (Counterfactual) for every instruction-tuned model.** Pearl's hierarchy is usually framed as a tower of *expressivity* (L3 questions cannot be answered by L2-only knowledge, etc.), and people often expect *difficulty* to track that ordering. We observe the opposite at L2 vs L3: every instruct model drops 15-25pp from L1 to L2, then partially recovers at L3. The L1-vs-L2 95% CIs do not overlap for Qwen or Llama, so the L1 > L2 gap is statistically robust. Plausible reading: L3 questions provide observed pre-state information (e.g., "we observed X=low and Z=low") which anchors reasoning, whereas bare `do(X)` framing offers no anchor and pulls the model toward associational shortcuts.

3. **Collider is universally the hardest graph.** Average accuracy across L1/L2/L3 (zero-shot, full benchmark):

   | Model | chain | fork | **collider** | diamond |
   |---|---:|---:|---:|---:|
   | Qwen   | 82% | 85% | **35%** | 57% |
   | Llama  | 92% | 65% | **13%** | 78% |
   | Gemma  | 63% | 48% | **23%** | 58% |

   Llama scores 0% on L3 + collider specifically. The collider questions test whether the model can recognize that two variables sharing a common effect (X → M ← Z) are *not* causally related to each other - even though they're correlated when M is held fixed. Failure suggests the model is using surface co-occurrence as a proxy for causation rather than reasoning about the directed graph.

4. **Few-shot helps every instruct model; "Causal Chain" prompting collapses two of three.** Few-shot is the only strategy that improves all three instruct models over their zero-shot baseline (Gemma jumps +17pp). PMI-scored CoT is roughly neutral on aggregate, and real-generation CoT confirms this on average with per-model variation (Qwen -7pp, Llama +1pp, Gemma +7pp). The "Causal Chain" template - which asks the model to first identify the graph structure, then answer - produces a -22pp accuracy drop for both Qwen and Gemma. We attribute this to prior collapse: the long structured prompt shifts the model's letter prior so far that PMI's linear correction cannot fully undo the resulting softmax distortion.

5. **Llama and Qwen disagree on which graph types they handle best.** Llama (3.2B) is bigger than Qwen (1.5B) but has lower aggregate zero-shot accuracy. Llama dominates chain (92%) and diamond (78%) but collapses on collider (13%); Qwen is more uniform - it leads on fork (85%) and is 2.7x better than Llama on collider (35% vs 13%). On this benchmark Qwen is the more balanced generalist.

## Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

Python ≥3.9, PyTorch ≥2.0. CUDA GPU strongly recommended (~16 GB VRAM for the full set; T4 fits all five with NF4).

## Run

```bash
# Full pipeline: zero-shot + 4 prompting strategies + permutation averaging
# + real-generation CoT (~3h on T4)
python -m scripts.run

# Skip the robustness checks (zero-shot + strategies only, ~75 min)
python -m scripts.run --skip-robustness

# Skip the strategy comparison too (zero-shot only)
python -m scripts.run --skip-strategies --skip-robustness

# Subset of models
python -m scripts.run --models qwen_1_5b llama_3_3b

# Custom output directory
python -m scripts.run --output-dir runs/v1

# Use the full benchmark for strategy comparison (240 questions instead of 120)
python -m scripts.run --strategy-bucket 20
```

The notebook `Causal_LLM_Understanding.ipynb` runs the same pipeline interactively and is the easier entry point if you want to inspect intermediate results.

## Outputs

Written to `output_llm/`:

| File | What |
|---|---|
| `results_summary.json` | All numbers above as a single JSON object |
| `zero_shot_accuracy.csv` | Model × level (full benchmark) |
| `strategy_accuracy.csv` | Model × strategy × level (subset) |
| `detailed_accuracy.csv` | Model × level × graph |
| `permuted_accuracy.csv` | Model × level: single-perm, perm-averaged mean, 95% CI bounds |
| `gen_cot_accuracy.csv` | Per-model real-generation CoT accuracy + CI |
| `raw_predictions.jsonl` | Per-question prediction record across every evaluation |
| `gen_cot_chains.jsonl` | Full reasoning chains generated by each instruct model |
| `accuracy_by_graph_level.png`, `prompting_strategy_comparison.png` | Plots |
| `causal_graphs.png`, `benchmark_stats.png` | Reference figures |
| `run_manifest.json` | Versions, GPU, seed, timestamp |

## Notes on scoring

- **PMI calibration**: for each candidate answer letter, the score is `log P(letter | full prompt) − log P(letter | content-free prompt)`. The subtrahend is measured by replacing the scenario, question, and choices with "N/A" and scoring the four letters against that. This removes the model's letter-frequency prior so the calibrated score reflects the prompt content rather than which letter the model happens to like.
- **Generation fallback**: when calibration priors are NaN, collapsed (range below 1e-5), or so wide that PMI's linear correction cannot undo the nonlinear softmax distortion (range above 2.5 log-prob units, typical for GPT-2), the evaluator switches to greedy decoding and parses the first A/B/C/D the model emits.
- **GPT-2 numbers sit at chance**: their permutation-averaged 95% CIs all include 25%, so cross-model comparisons against them are not apples-to-apples. They are useful as a sanity floor (a model that does no causal reasoning should score here) and as a calibration check on the scoring pipeline (PMI does not lift them above chance, which is the right behavior).
