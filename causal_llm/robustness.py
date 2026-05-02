"""Permutation-averaged accuracy + bootstrap CIs for letter-bias-robust scoring."""

from collections import defaultdict

import numpy as np
import torch

from .config import MODEL_LABELS
from .evaluator import evaluate_model

ABCD = ['A', 'B', 'C', 'D']


def _cyclic_perm(shift):
    return {ABCD[i]: ABCD[(i + shift) % 4] for i in range(4)}


# Four cyclic shifts of A/B/C/D. Across all four, the correct answer letter
# hits each of the four positions exactly once.
PERMUTATIONS = [_cyclic_perm(k) for k in range(4)]


def apply_permutation(q, perm):
    """Return a copy of q with choices and answer relabelled per perm.

    perm maps OLD letter -> NEW position. The choice text that was at old
    letter L now lives at new position perm[L]; the correct answer letter
    moves from q['answer'] to perm[q['answer']].
    """
    new_choices = {perm[L]: q['choices'][L] for L in ABCD}
    return {**q, 'choices': new_choices, 'answer': perm[q['answer']]}


def bootstrap_ci(values, n_iter=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the mean. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot = np.empty(n_iter)
    for i in range(n_iter):
        boot[i] = arr[rng.integers(0, n, n)].mean()
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def permutation_averaged_zero_shot(active_models, models, tokenizers, sample, device):
    """Score each question 4 times (one per cyclic perm) and average per question.

    Returns:
        permuted_per_question[(model_key, level)] = {qid: mean_correct over 4 perms}
    """
    out = {}
    for mk in active_models:
        print(f"\n{'=' * 60}")
        print(f"Permutation-averaged zero-shot: {MODEL_LABELS[mk]}")
        print('=' * 60)
        for level in ['L1', 'L2', 'L3']:
            questions = [q for q in sample if q['level'] == level]
            per_q_correct = defaultdict(list)
            for k, perm in enumerate(PERMUTATIONS):
                permuted = [apply_permutation(q, perm) for q in questions]
                results, _ = evaluate_model(
                    mk, permuted, 'zero_shot', models, tokenizers, device,
                    verbose=(k == 0),
                )
                for r in results:
                    per_q_correct[r['id']].append(int(r['correct']))
            per_q_acc = {qid: float(np.mean(flags)) for qid, flags in per_q_correct.items()}
            out[(mk, level)] = per_q_acc
            mean_acc = float(np.mean(list(per_q_acc.values()))) if per_q_acc else 0.0
            print(f"  {level}: permutation-averaged acc = {mean_acc:.1%} "
                  f"({len(per_q_acc)} questions)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def compute_bootstrap_summary(permuted_per_question, single_perm_baseline):
    """Compute mean + 95% CI per (model, level) cell.

    Args:
        permuted_per_question: output of `permutation_averaged_zero_shot`.
        single_perm_baseline: function (mk, level) -> single-perm accuracy.
            Pass `lambda mk, lv: strategy_results[(mk, 'zero_shot')]['by_level'][lv]`
            to compare against the strategy-table zero-shot row.

    Returns:
        bootstrap_summary[(model_key, level)] = {mean, lo, hi, single_perm, n}
    """
    summary = {}
    for (mk, level), per_q in permuted_per_question.items():
        values = list(per_q.values())
        mean, lo, hi = bootstrap_ci(values)
        summary[(mk, level)] = {
            'mean': mean,
            'lo': lo,
            'hi': hi,
            'single_perm': float(single_perm_baseline(mk, level)),
            'n': len(values),
        }
    return summary


def print_robustness_summary(active_models, bootstrap_summary):
    """Print the permutation-averaged accuracy table with bootstrap CIs."""
    print(f"\n{'=' * 86}")
    print("Permutation-averaged accuracy + bootstrap 95% CIs")
    print('=' * 86)
    header = (f"{'Model':25s} | {'Level':5s} | "
              f"{'Single':>8s} | {'Perm':>8s} | {'95% CI':>17s} | {'Δ':>6s}")
    print(header)
    print('-' * len(header))
    for mk in active_models:
        for level in ['L1', 'L2', 'L3']:
            s = bootstrap_summary.get((mk, level))
            if s is None:
                continue
            delta = s['mean'] - s['single_perm']
            print(f"{MODEL_LABELS[mk]:25s} | {level:5s} | "
                  f"{s['single_perm']:>7.1%} | {s['mean']:>7.1%} | "
                  f"[{s['lo']:>5.1%}, {s['hi']:>5.1%}] | {delta:>+5.1%}")

    print('\nCells whose 95% CI includes the random baseline (25%):')
    any_at_chance = False
    for (mk, level), s in bootstrap_summary.items():
        if s['lo'] <= 0.25 <= s['hi']:
            print(f"  {MODEL_LABELS[mk]:25s} {level}: "
                  f"{s['mean']:.1%} [{s['lo']:.1%}, {s['hi']:.1%}]")
            any_at_chance = True
    if not any_at_chance:
        print('  (none)')
