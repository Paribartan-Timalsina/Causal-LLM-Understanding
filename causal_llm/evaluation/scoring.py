"""Answer scoring via log-probability and greedy generation."""

import math

import torch
import torch.nn.functional as F

from ..config import MAX_SEQ_LEN
from .prompts import CONTENT_FREE_PROMPTS


@torch.no_grad()
def score_answer_logprob(model, tokenizer, prompt, device, max_len=MAX_SEQ_LEN,
                         warn_truncation=True):
    """Forced-completion scoring.

    For each candidate letter (A/B/C/D), compute the conditional log-likelihood
    of the completion string " <LETTER>" appended to the prompt.

    Returns:
        dict: {'A': logprob, 'B': logprob, 'C': logprob, 'D': logprob}
    """
    model.eval()
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    scores = {}
    for letter in ["A", "B", "C", "D"]:
        completion = f" {letter}"
        full_ids = tokenizer(prompt + completion, add_special_tokens=False).input_ids
        completion_ids = full_ids[len(prompt_ids):]

        if len(completion_ids) == 0:
            sep_ids = tokenizer(completion, add_special_tokens=False).input_ids
            completion_ids = full_ids[-len(sep_ids):]

        if len(completion_ids) == 0:
            raise ValueError(f"Empty completion tokenization for letter={letter}.")

        truncated = False
        if len(full_ids) > max_len:
            truncated = True
            start = len(full_ids) - max_len
            full_ids_trunc = full_ids[start:]
        else:
            full_ids_trunc = full_ids

        completion_start = len(full_ids_trunc) - len(completion_ids)
        if completion_start <= 0:
            raise ValueError(
                "Prompt too long: completion is not conditionable (completion_start<=0). "
                "Increase max_len or shorten prompts."
            )

        input_ids = torch.tensor([full_ids_trunc], device=device)
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

        total_lp = 0.0
        for j, tok in enumerate(full_ids_trunc[completion_start:]):
            pos = completion_start + j
            prev_pos = pos - 1
            logits_f = logits[prev_pos].float()
            logits_f = torch.clamp(logits_f, min=-1e4, max=1e4)
            logits_f = torch.nan_to_num(logits_f, nan=0.0, posinf=1e4, neginf=-1e4)
            lp = F.log_softmax(logits_f, dim=-1)[tok].item()
            total_lp += lp

        scores[letter] = float(total_lp)

        if truncated and warn_truncation:
            warn_truncation = False
            print(
                f"  Warning: prompt was truncated to {max_len} tokens for scoring. "
                f"(full_len={len(full_ids)})"
            )

    return scores


@torch.no_grad()
def score_answer_generation(model, tokenizer, prompt, device, max_len=MAX_SEQ_LEN,
                            max_new_tokens=5):
    """Greedy-decode a short continuation and return the first A/B/C/D letter found."""
    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_len
    ).input_ids.to(device)
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    for ch in generated.strip():
        if ch.upper() in "ABCD":
            return ch.upper()
    return None


@torch.no_grad()
def compute_calibration_priors(model, tokenizer, strategy, device, max_len=MAX_SEQ_LEN):
    """Score answer letters against a content-free prompt to measure token prior bias.

    Returns:
        dict: {'A': logprob, 'B': logprob, ...} to subtract from raw scores (PMI).
    """
    cf_prompt = CONTENT_FREE_PROMPTS[strategy]
    return score_answer_logprob(model, tokenizer, cf_prompt, device,
                                max_len=max_len, warn_truncation=False)
