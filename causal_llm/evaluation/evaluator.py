"""Main evaluation pipeline with PMI calibration and fallback."""

import math
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm

from ..config import MODEL_LABELS, MODEL_MAX_SEQ_LEN, MAX_SEQ_LEN, PROMPTING_STRATEGIES
from .prompts import PROMPT_FORMATTERS
from .scoring import compute_calibration_priors, score_answer_logprob, score_answer_generation


def evaluate_model(model_key, model, tokenizer, questions, strategy="zero_shot",
                   device=None, verbose=True):
    """Evaluate a model on a list of questions using a prompting strategy.

    Applies PMI calibration: subtracts content-free log-probs to remove
    the model's inherent token-prior bias toward certain letters.

    Falls back to greedy-generation if calibrated log-prob scoring is degenerate.

    Returns:
        tuple: (results_list, accuracy)
    """
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    formatter = PROMPT_FORMATTERS[strategy]
    max_len = MODEL_MAX_SEQ_LEN.get(model_key, MAX_SEQ_LEN)
    label = MODEL_LABELS.get(model_key, model_key)

    calibration = compute_calibration_priors(model, tokenizer, strategy, device,
                                             max_len=max_len)

    use_generation_fallback = False
    if any(v != v for v in calibration.values()):
        print(f"  WARNING: NaN calibration for {label} -- using generation fallback.")
        use_generation_fallback = True
        calibration = {k: 0.0 for k in calibration}
    elif not all(math.isfinite(v) for v in calibration.values()):
        print(f"  WARNING: Non-finite calibration for {label} -- using generation fallback.")
        use_generation_fallback = True
        calibration = {k: 0.0 for k in calibration}

    if verbose:
        print(f"  Calibration priors ({strategy}): "
              + ", ".join(f"{k}={v:.3f}" for k, v in calibration.items()))
        if use_generation_fallback:
            print(f"  >> Falling back to greedy-generation answer extraction")

    results = []
    correct = 0
    desc = f"{label} [{strategy}]"

    for q in tqdm(questions, desc=desc, leave=False):
        prompt = formatter(q)

        if use_generation_fallback:
            gen_answer = score_answer_generation(model, tokenizer, prompt, device,
                                                 max_len=max_len)
            prediction = gen_answer if gen_answer is not None else "A"
            scores = {l: 0.0 for l in "ABCD"}
            raw_scores = scores
        else:
            raw_scores = score_answer_logprob(model, tokenizer, prompt, device,
                                              max_len=max_len)
            scores = {l: raw_scores[l] - calibration[l] for l in raw_scores}
            prediction = max(scores, key=scores.get)

        is_correct = prediction.strip().upper() == q["answer"].strip().upper()
        correct += int(is_correct)

        results.append({
            "id": q["id"],
            "graph": q["graph"],
            "level": q["level"],
            "answer": q["answer"],
            "prediction": prediction,
            "correct": is_correct,
            "scores": scores,
            "raw_scores": raw_scores,
            "prompt": prompt,
        })

    accuracy = correct / len(questions) if questions else 0
    if verbose:
        print(f"  {desc}: {accuracy:.1%} ({correct}/{len(questions)})")

    return results, accuracy


def run_full_evaluation(loaded_models, loaded_tokenizers, questions, strategies=None,
                        device=None, verbose=True):
    """Run evaluation across all models and strategies.

    Args:
        loaded_models: dict mapping model_key -> model
        loaded_tokenizers: dict mapping model_key -> tokenizer
        questions: list of question dicts
        strategies: list of strategy names (defaults to PROMPTING_STRATEGIES)
        device: torch device
        verbose: print progress

    Returns:
        dict: Mapping (model_key, strategy) -> {'results': list, 'accuracy': float}
    """
    if strategies is None:
        strategies = PROMPTING_STRATEGIES

    all_results = {}
    for model_key in loaded_models:
        model = loaded_models[model_key]
        tokenizer = loaded_tokenizers[model_key]

        if verbose:
            label = MODEL_LABELS.get(model_key, model_key)
            print(f"\n{'=' * 60}")
            print(f"Evaluating {label}")
            print(f"{'=' * 60}")

        for strategy in strategies:
            results, accuracy = evaluate_model(
                model_key, model, tokenizer, questions,
                strategy=strategy, device=device, verbose=verbose,
            )
            all_results[(model_key, strategy)] = {
                "results": results,
                "accuracy": accuracy,
            }

    return all_results
