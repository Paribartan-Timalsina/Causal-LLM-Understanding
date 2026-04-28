"""Forced-completion log-prob scoring + PMI calibration."""

import math

import torch
import torch.nn.functional as F

from .config import MAX_SEQ_LEN
from .prompts import CONTENT_FREE_PROMPTS


@torch.no_grad()
def score_answer_logprob(model, tokenizer, prompt, device, max_len=MAX_SEQ_LEN,
                         warn_truncation=True):
    """For each candidate letter, return log P(' <LETTER>' | prompt)."""
    model.eval()
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    scores = {}

    for letter in ['A', 'B', 'C', 'D']:
        completion = f' {letter}'
        full_ids = tokenizer(prompt + completion, add_special_tokens=False).input_ids

        # BPE may merge the last prompt token with the first completion token.
        # In that case naive prefix slicing returns wrong ids; fall back to
        # tokenizing the completion alone and grabbing the suffix.
        if full_ids[: len(prompt_ids)] == prompt_ids:
            completion_ids = full_ids[len(prompt_ids):]
        else:
            sep_ids = tokenizer(completion, add_special_tokens=False).input_ids
            completion_ids = full_ids[-len(sep_ids):] if sep_ids else []

        if not completion_ids:
            raise ValueError(f'Empty completion tokenization for letter={letter}.')

        truncated = len(full_ids) > max_len
        full_ids_trunc = full_ids[len(full_ids) - max_len:] if truncated else full_ids
        completion_start = len(full_ids_trunc) - len(completion_ids)
        if completion_start <= 0:
            raise ValueError('Prompt too long: completion is not conditionable.')

        input_ids = torch.tensor([full_ids_trunc], device=device)
        logits = model(input_ids=input_ids).logits[0]

        # NB: don't clamp logits before log_softmax -- 4-bit quantized models
        # can produce extreme values and clamping flattens the softmax,
        # collapsing all letter scores to be identical. log_softmax is
        # numerically stable; just sanitize NaN/inf.
        total_lp = 0.0
        for j, tok in enumerate(full_ids_trunc[completion_start:]):
            prev = completion_start + j - 1
            row = torch.nan_to_num(
                logits[prev].float(),
                nan=0.0, posinf=float('inf'), neginf=float('-inf'),
            )
            lp = F.log_softmax(row, dim=-1)[tok].item()
            total_lp += lp if math.isfinite(lp) else -1e4

        scores[letter] = float(total_lp)

        if truncated and warn_truncation:
            warn_truncation = False
            print(f'  Warning: prompt truncated to {max_len} tokens (full_len={len(full_ids)})')

    return scores


@torch.no_grad()
def score_answer_generation(model, tokenizer, prompt, device, max_len=MAX_SEQ_LEN,
                            max_new_tokens=5):
    """Greedy decode and return the first A/B/C/D letter, or None."""
    input_ids = tokenizer(
        prompt, return_tensors='pt', truncation=True, max_length=max_len,
    ).input_ids.to(device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    for ch in text.strip():
        if ch.upper() in 'ABCD':
            return ch.upper()
    return None


def compute_calibration_priors(model, tokenizer, strategy, device, max_len=MAX_SEQ_LEN):
    """Score letters against the content-free prompt to measure letter bias."""
    return score_answer_logprob(
        model, tokenizer, CONTENT_FREE_PROMPTS[strategy], device,
        max_len=max_len, warn_truncation=False,
    )
