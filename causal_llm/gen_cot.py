"""Real-generation chain-of-thought: actually generate the chain, then parse the letter."""

import re

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import MAX_SEQ_LEN, MODEL_LABELS, MODEL_MAX_SEQ_LEN
from .prompts import format_chain_of_thought
from .robustness import bootstrap_ci

DEFAULT_INSTRUCT_MODELS = ['qwen_1_5b', 'llama_3_3b', 'gemma_2_2b']
DEFAULT_MAX_NEW_TOKENS = 250

# Explicit answer-pattern matches are preferred over the standalone-letter
# fallback. Each is tried in order; the LAST match per pattern wins so
# "I considered A but the answer is C" yields C.
_ANSWER_PATTERNS = [
    re.compile(r'(?:final\s+answer|the\s+answer)\s*(?:is|:)?\s*\*{0,2}\s*\(?([ABCD])\)?', re.IGNORECASE),
    re.compile(r'answer\s+is\s*\*{0,2}\s*\(?([ABCD])\)?', re.IGNORECASE),
    re.compile(r'answer\s*[:=]\s*\*{0,2}\s*\(?([ABCD])\)?', re.IGNORECASE),
    re.compile(r'\*\*\s*([ABCD])\s*\*\*'),
]
# Fallback: take the FIRST standalone A/B/C/D character in the text.
# First, not last, because models often answer cleanly at the start
# (" C Explanation: ...") and then mention A/B/C/D as variable references.
_FALLBACK_LETTER = re.compile(r'(?:^|[^A-Za-z])([ABCD])(?=[^A-Za-z]|$)')


def extract_answer_letter(text):
    """Return the model's chosen letter, or None if undetectable."""
    if not text:
        return None
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            return matches[-1].upper()
    matches = _FALLBACK_LETTER.findall(text)
    if matches:
        return matches[0].upper()
    return None


@torch.no_grad()
def generate_cot_response(model, tokenizer, prompt, device, max_len,
                          max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Greedy decode and return the generated text (without the prompt)."""
    enc = tokenizer(
        prompt, return_tensors='pt', truncation=True, max_length=max_len,
    ).to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    output = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    new_tokens = output[0][enc.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_gen_cot(model_key, sample, models, tokenizers, device,
                     prompt_formatter=format_chain_of_thought,
                     max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
    """Run real-generation CoT on one model.

    Returns dict with keys: accuracy, ci, by_level, extract_rate, raw.
        raw is a list of per-question dicts with the full generated text.
    """
    model = models[model_key]
    tokenizer = tokenizers[model_key]
    max_len = MODEL_MAX_SEQ_LEN.get(model_key, MAX_SEQ_LEN)

    raw = []
    correct_flags = []
    extracted_count = 0

    for q in tqdm(sample, desc=MODEL_LABELS.get(model_key, model_key), leave=False):
        prompt = prompt_formatter(q)
        text = generate_cot_response(model, tokenizer, prompt, device,
                                     max_len, max_new_tokens)
        letter = extract_answer_letter(text)
        if letter is not None:
            extracted_count += 1
        is_correct = (letter is not None and letter == q['answer'])
        correct_flags.append(int(is_correct))
        raw.append({
            'qid': q['id'],
            'graph': q['graph'],
            'level': q['level'],
            'gold': q['answer'],
            'prediction': letter,
            'correct': is_correct,
            'generation': text,
        })

    accuracy = float(np.mean(correct_flags)) if correct_flags else 0.0
    mean, lo, hi = bootstrap_ci(correct_flags)
    by_level = {
        lv: (float(np.mean([r['correct'] for r in raw if r['level'] == lv]))
             if any(r['level'] == lv for r in raw) else 0.0)
        for lv in ['L1', 'L2', 'L3']
    }
    extract_rate = extracted_count / len(raw) if raw else 0.0

    return {
        'accuracy': accuracy,
        'ci': {'mean': mean, 'lo': lo, 'hi': hi},
        'by_level': by_level,
        'extract_rate': extract_rate,
        'raw': raw,
    }


def run_gen_cot(active_models, models, tokenizers, sample, device,
                instruct_only=True):
    """Run gen-CoT on every active model (or only the instruct ones).

    Returns gen_cot_results[mk] = the dict from evaluate_gen_cot.
    """
    targets = [mk for mk in active_models
               if (not instruct_only) or mk in DEFAULT_INSTRUCT_MODELS]

    print(f"Gen-CoT sample: {len(sample)} questions, "
          f"{len(targets)} model{'s' if len(targets) != 1 else ''}.")
    print(f"  Models: {[MODEL_LABELS[m] for m in targets]}")

    out = {}
    for mk in targets:
        print(f"\n{'=' * 60}")
        print(f"Generation-based CoT: {MODEL_LABELS[mk]}")
        print('=' * 60)
        result = evaluate_gen_cot(mk, sample, models, tokenizers, device)
        out[mk] = result
        ci = result['ci']
        bl = result['by_level']
        print(f"  Letter extracted: {result['extract_rate']:.0%} of generations")
        print(f"  Accuracy: {result['accuracy']:.1%}  "
              f"95% CI [{ci['lo']:.1%}, {ci['hi']:.1%}]")
        print(f"  By level: L1={bl['L1']:.1%}  L2={bl['L2']:.1%}  L3={bl['L3']:.1%}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def print_gen_cot_comparison(gen_cot_results, strategy_results):
    """Side-by-side: PMI-CoT (from strategy_results) vs real Gen-CoT."""
    print(f"\n{'=' * 82}")
    print("CoT comparison: PMI-scoring vs real generation")
    print('=' * 82)
    header = (f"{'Model':25s} | {'PMI-CoT':>8s} | {'Gen-CoT':>8s} | "
              f"{'95% CI (gen)':>17s} | {'Δ':>6s}")
    print(header)
    print('-' * len(header))
    for mk, result in gen_cot_results.items():
        pmi = strategy_results.get((mk, 'chain_of_thought'), {}).get('accuracy', 0.0)
        gen = result['accuracy']
        ci = result['ci']
        delta = gen - pmi
        print(f"{MODEL_LABELS[mk]:25s} | "
              f"{pmi:>7.1%} | {gen:>7.1%} | "
              f"[{ci['lo']:>5.1%}, {ci['hi']:>5.1%}] | {delta:>+5.1%}")
