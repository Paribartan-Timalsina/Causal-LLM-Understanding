# Can LLMs Reason Causally? Probing Causal Inference Across Pearl's Hierarchy

A systematic evaluation of causal reasoning capabilities in Large Language Models, testing whether LLMs perform genuine causal inference or merely exploit statistical shortcuts.

## Overview

This project investigates how LLM causal reasoning ability varies across:
- **Pearl's Causal Hierarchy** (Association, Intervention, Counterfactual)
- **Model scale** (124M to 2.7B parameters)
- **Prompting strategies** (Zero-shot, Few-shot, Chain-of-Thought, Causal Chain)

### Models Evaluated

| Model | Parameters | Type | Quantization |
|-------|-----------|------|-------------|
| GPT-2 Small | 124M | Base (autoregressive) | FP32 |
| GPT-2 Large | 774M | Base (autoregressive) | FP32 |
| Phi-2 | 2.7B | Instruction-tuned | 4-bit (NF4) |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Causal LLM understanding"

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- Transformers >= 4.35
- CUDA-capable GPU recommended (16 GB VRAM for all models)

## Quick Start

### 1. Run Evaluation

```bash
# Evaluate all models with zero-shot prompting
python scripts/run_evaluation.py

# Evaluate specific models and strategies
python scripts/run_evaluation.py \
    --models gpt2_small gpt2_large \
    --strategies zero_shot chain_of_thought \
    --num-questions 20 \
    --output-dir ./output_llm
```

### 2. Run Internal Analysis

```bash
# Linear probing and attention analysis
python scripts/run_analysis.py \
    --models gpt2_small gpt2_large \
    --num-samples 200
```

### 3. Generate Report

```bash
# Generate all visualizations
python scripts/generate_report.py --output-dir ./output_llm
```

### 4. Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Or run individually
python tests/test_benchmark.py
python tests/test_prompts.py
python tests/test_scoring.py
```

## Benchmark Design

### Five Canonical Causal Structures

| Structure | Graph | Key Property |
|-----------|-------|-------------|
| **Chain** | X -> Y -> Z | Mediation: X causes Z only through Y |
| **Fork** | X <- C -> Z | Common cause: X and Z correlated but non-causal |
| **Collider** | X -> M <- Z | Explaining away: conditioning on M creates spurious dependence |
| **Diamond** | X -> {Y,Z} -> W | Multiple pathways from X to W |
| **Instrument** | Z -> X -> Y, U -> {X,Y} | Instrumental variable for causal identification |

### Three Levels of Causal Reasoning (Pearl's Hierarchy)

| Level | Name | Question Type | Requires |
|-------|------|---------------|----------|
| L1 | **Association** | P(Y \| X) | Passive observation |
| L2 | **Intervention** | P(Y \| do(X)) | Manipulating variables |
| L3 | **Counterfactual** | P(Y_x \| X', Y') | Imagining alternatives |

### Evaluation Methodology

- **PMI Calibration**: Content-free prompt calibration removes token-prior bias
- **Log-probability scoring**: Uniform across base and instruction-tuned models
- **Generation fallback**: Greedy decoding when log-prob scoring is degenerate