"""Per-model evaluation loop with PMI / generation-fallback scoring routing."""

from __future__ import annotations

import math

import torch
from tqdm.auto import tqdm

from .config import MAX_SEQ_LEN, MODEL_LABELS, MODEL_MAX_SEQ_LEN
from .prompts import PROMPT_FORMATTERS
from .scoring import (
    compute_calibration_priors,
    score_answer_generation,
    score_answer_logprob,
)

# PMI calibration thresholds (in log-prob units).
_DEGEN_RANGE = 1e-5     # priors fully collapsed -> generation fallback
_RAW_RANGE = 1e-2       # priors effectively uniform -> skip PMI subtraction
_LARGE_PRIOR = 2.5      # base LMs with this much prior bias -> generation
_SMALL_BASE = {'gpt2_small', 'gpt2_large'}


def _pick_scoring_mode(priors: dict[str, float], model_key: str) -> tuple[str, str | None]:
    """Decide how to score this (model, strategy): pmi / raw / generation."""
    values = list(priors.values())
    if any(v != v for v in values):
        return 'generation', 'NaN calibration priors'
    if not all(math.isfinite(v) for v in values):
        return 'generation', 'non-finite calibration priors'
    rng = max(values) - min(values)
    if rng < _DEGEN_RANGE:
        return 'generation', f'priors fully collapsed (range={rng:.2e})'
    if rng < _RAW_RANGE:
        return 'raw_logprob', f'priors uniform (range={rng:.2e}); skipping PMI'
    if model_key in _SMALL_BASE and rng > _LARGE_PRIOR:
        return 'generation', f'base LM with large prior range ({rng:.2f} > {_LARGE_PRIOR})'
    return 'pmi_logprob', None


def _format_prior(v: float) -> str:
    if v != v:
        return 'nan'
    if not math.isfinite(v):
        return '+inf' if v > 0 else '-inf'
    return f'{v:.6f}'


def evaluate_model(
    model_key: str,
    questions: list[dict],
    strategy: str,
    models: dict,
    tokenizers: dict,
    device: torch.device,
    verbose: bool = True,
) -> tuple[list[dict], float]:
    """Score every question for one (model, strategy) pair.

    Routing logic for scoring mode:
      pmi_logprob  - default. Subtracts content-free prior from raw logprob.
      raw_logprob  - prior is effectively uniform; subtraction is just noise.
      generation   - greedy decode and extract first A/B/C/D letter
                     (used when priors are NaN/collapsed, or when a base LM
                     has a prior range so large that PMI's linear correction
                     can't undo the nonlinear softmax bias).

    Returns (per-question result dicts, accuracy).
    """
    model = models[model_key]
    tokenizer = tokenizers[model_key]
    formatter = PROMPT_FORMATTERS[strategy]
    max_len = MODEL_MAX_SEQ_LEN.get(model_key, MAX_SEQ_LEN)

    priors = compute_calibration_priors(model, tokenizer, strategy, device, max_len=max_len)
    mode, reason = _pick_scoring_mode(priors, model_key)

    if verbose:
        prior_str = ', '.join(f'{k}={_format_prior(v)}' for k, v in priors.items())
        rng = max(priors.values()) - min(priors.values()) if all(math.isfinite(v) for v in priors.values()) else float('nan')
        rng_str = f' | range={rng:.2e}' if math.isfinite(rng) else ''
        print(f'  Calibration priors ({strategy}): {prior_str}{rng_str}')
        print(f'  Scoring mode: {mode}' + (f'  ({reason})' if reason else ''))

    use_generation = (mode == 'generation')
    # In raw_logprob mode, calibration is a no-op (subtract zeros).
    calibration = priors if mode == 'pmi_logprob' else {k: 0.0 for k in priors}

    results: list[dict] = []
    correct = 0
    label = MODEL_LABELS.get(model_key, model_key)
    desc = f'{label} [{strategy}]'

    for q in tqdm(questions, desc=desc, leave=False):
        prompt = formatter(q)
        if use_generation:
            answer = score_answer_generation(model, tokenizer, prompt, device, max_len=max_len)
            prediction = answer if answer is not None else 'A'
            scores = {l: 0.0 for l in 'ABCD'}
            raw_scores = scores
        else:
            raw_scores = score_answer_logprob(model, tokenizer, prompt, device, max_len=max_len)
            scores = {l: raw_scores[l] - calibration[l] for l in raw_scores}
            prediction = max(scores, key=lambda k: scores[k])

        is_correct = prediction.strip().upper() == q['answer'].strip().upper()
        correct += int(is_correct)
        results.append({
            'id': q['id'],
            'graph': q['graph'],
            'level': q['level'],
            'answer': q['answer'],
            'prediction': prediction,
            'correct': is_correct,
            'scores': scores,
            'raw_scores': raw_scores,
            'scoring_mode': mode,
        })

    accuracy = correct / len(questions) if questions else 0.0
    if verbose:
        print(f'  {desc}: {accuracy:.1%} ({correct}/{len(questions)})')
    return results, accuracy
