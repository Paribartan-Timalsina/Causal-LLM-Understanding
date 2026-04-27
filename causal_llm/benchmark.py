"""Benchmark generation: multiple-choice causal reasoning questions.

The benchmark tests Pearl's three reasoning levels:
- L1 (Association): observation-based
- L2 (Intervention): do-calculus
- L3 (Counterfactual): would-have-been

across four canonical graphs: chain, fork, collider, diamond.

Ground-truth answers are derived from d-separation rules and SCM algebra.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Iterable

from .config import NUM_QUESTIONS_PER_CELL, SEED
from .scenarios import SCENARIOS

Question = dict  # alias for readability

_L1_OBSERVE_PHRASES = [
    lambda v: f'We observe that {v} is high. What do we expect for',
    lambda v: f'Suppose we notice that {v} is at a high level. What should we predict for',
    lambda v: f'Given that {v} is observed to be high, what is the likely outcome for',
    lambda v: f'If we see that {v} is high, what can we infer about',
    lambda v: f'Imagine we measure {v} and find it is high. What follows for',
]
_L2_INTERVENE_PHRASES = [
    lambda v: f'If we intervene and force {v} to be high (do({v}=high)), what happens to',
    lambda v: f'Suppose we externally set {v} to a high value. What is the effect on',
    lambda v: f'If we perform an intervention do({v}=high), what change do we expect in',
    lambda v: f'Consider forcibly making {v} high, breaking its natural causes. What happens to',
    lambda v: f'If an experimenter manipulates {v} to be high, what is the causal impact on',
]
_L3_CF_PHRASES = [
    lambda x, z: f'We observed {x}=low and {z}=low. If {x} had been high instead, what would {z} have been?',
    lambda x, z: f'Given that {x} was low and {z} was low, suppose {x} had been high. What would {z} be?',
    lambda x, z: f'In a world where {x}=low led to {z}=low, what would {z} have been if {x} were high?',
    lambda x, z: f'Counterfactual: we saw {x}=low, {z}=low. Had {x} been high, what about {z}?',
    lambda x, z: f'{x} was low and {z} ended up low. If we could go back and set {x} to high, what would {z} be?',
]
_CONTEXT_PREFIXES = [
    '',
    'Consider this situation carefully. ',
    'Think about the causal relationships here. ',
    'Pay attention to the direction of causation. ',
    'Analyze the following causal scenario. ',
]


def make_question(scenario: dict, graph_type: str, level: str, q_id: int) -> Question:
    """Build a single question dict with a ground-truth answer.

    `q_id` selects phrasing variants so repeating the same scenario produces
    different surface text.
    """
    v = scenario['vars']
    story = _CONTEXT_PREFIXES[q_id % len(_CONTEXT_PREFIXES)] + scenario['story']

    if graph_type == 'chain':
        X, Y, Z = v['X'], v['Y'], v['Z']
        if level == 'L1':
            qtext = _L1_OBSERVE_PHRASES[q_id % len(_L1_OBSERVE_PHRASES)](X) + f' {Z}?'
            return _q(q_id, graph_type, level, story, qtext, {
                'A': f'{Z} is likely higher than average',
                'B': f'{Z} is likely lower than average',
                'C': f'{Z} is unaffected',
                'D': 'Cannot determine from the given information',
            }, 'A',
                f'In a chain X->Y->Z, observing X=high implies Y=high implies Z=high.')
        if level == 'L2':
            qtext = _L2_INTERVENE_PHRASES[q_id % len(_L2_INTERVENE_PHRASES)](X) + f' {Z}?'
            return _q(q_id, graph_type, level, story, qtext, {
                'A': f'{Z} increases (causal effect propagates through {Y})',
                'B': f'{Z} is unaffected because the intervention breaks the chain',
                'C': f'{Z} decreases due to the intervention',
                'D': 'The effect depends on unobserved confounders',
            }, 'A',
                'Intervening on X in a chain still propagates: do(X=high)->Y=high->Z=high.')
        # L3
        qtext = _L3_CF_PHRASES[q_id % len(_L3_CF_PHRASES)](X, Z)
        return _q(q_id, graph_type, level, story, qtext, {
            'A': f'{Z} would have been higher',
            'B': f'{Z} would have remained low',
            'C': f'{Z} would have been lower',
            'D': 'Cannot determine without knowing the exact structural equations',
        }, 'A',
            'Counterfactual: setting X=high would propagate through Y to make Z higher.')

    if graph_type == 'fork':
        C, X, Z = v['C'], v['X'], v['Z']
        if level == 'L1':
            qtext = _L1_OBSERVE_PHRASES[q_id % len(_L1_OBSERVE_PHRASES)](X) + f' {Z}?'
            return _q(q_id, graph_type, level, story, qtext, {
                'A': f'{Z} is likely higher than average',
                'B': f'{Z} is likely lower than average',
                'C': f'{Z} is completely unaffected by {X}',
                'D': f'{Z} is inversely related to {X}',
            }, 'A',
                'In a fork C->X, C->Z: observing X=high suggests C=high, which suggests Z=high.')
        if level == 'L2':
            qtext = _L2_INTERVENE_PHRASES[q_id % len(_L2_INTERVENE_PHRASES)](X) + f' {Z}?'
            return _q(q_id, graph_type, level, story, qtext, {
                'A': f'{Z} increases because {X} and {Z} are correlated',
                'B': f'{Z} is unaffected because do({X}) severs the link from {C} to {X}',
                'C': f'{Z} decreases due to a compensatory mechanism',
                'D': f'{Z} doubles because of the intervention',
            }, 'B',
                'Intervening do(X=high) cuts the edge C->X. Since X does not cause Z (only C does), Z is unaffected.')
        # L3
        qtext = _L3_CF_PHRASES[q_id % len(_L3_CF_PHRASES)](X, Z)
        return _q(q_id, graph_type, level, story, qtext, {
            'A': f'{Z} would have been high because {X} directly causes {Z}',
            'B': f'{Z} would still be low because {Z} is caused by {C} (which stays low), not {X}',
            'C': f'{Z} would have been average',
            'D': 'Cannot determine',
        }, 'B',
            'Counterfactual do(X=high): Z is caused by C, not X. C=low is unchanged, so Z remains low.')

    if graph_type == 'collider':
        X, Z, M = v['X'], v['Z'], v['M']
        l1_phrases = [
            f'In the general population (without conditioning on {M}), if we observe {X} is high, what do we expect for {Z}?',
            f'Without selecting on {M}, suppose we see {X} is high. What can we say about {Z}?',
            f'Ignoring {M} entirely, we notice {X} is high. Does that tell us anything about {Z}?',
            f'Before knowing anything about {M}, we observe {X} is high. What follows for {Z}?',
            f'If we simply observe {X}=high in the population (without conditioning on {M}), what about {Z}?',
        ]
        l2_phrases = [
            f'We intervene and set do({X}=low). What is the *causal* effect of this intervention on {Z} (i.e., how does {Z} itself change as a result of the intervention)?',
            f'If we forcibly set {X} to low via do({X}=low), what is the causal effect on {Z}?',
            f'Suppose we perform do({X}=low). What is the causal impact on {Z} (independent of any conditioning on {M})?',
            f'What is the causal effect of intervening do({X}=low) on {Z}?',
            f'If an experimenter sets do({X}=low), what is the causal effect on {Z}?',
        ]
        if level == 'L1':
            return _q(q_id, graph_type, level, story, l1_phrases[q_id % len(l1_phrases)], {
                'A': f'{Z} is likely higher',
                'B': f'{Z} is likely lower',
                'C': f'{Z} is unrelated to {X} (no information gained)',
                'D': f'{Z} depends on {M}',
            }, 'C',
                'In a collider X->M<-Z, X and Z are marginally independent.')
        if level == 'L2':
            return _q(q_id, graph_type, level, story, l2_phrases[q_id % len(l2_phrases)], {
                'A': f"{Z} increases because removing {X}'s contribution means {Z} must explain {M}",
                'B': f'{Z} is unaffected because {X} does not cause {Z}',
                'C': f'{Z} decreases proportionally',
                'D': 'The question is ill-defined',
            }, 'B',
                'do(X) has no causal effect on Z; there is no causal path X->Z.')
        # L3
        qtext = _L3_CF_PHRASES[q_id % len(_L3_CF_PHRASES)](X, Z)
        return _q(q_id, graph_type, level, story, qtext, {
            'A': f'{Z} would have been higher to compensate and keep {M} positive',
            'B': f'{Z} would have remained low (no causal connection)',
            'C': f'{Z} would have been lower',
            'D': f'{M} would have changed but {Z} stays the same',
        }, 'B',
            'Counterfactual: X does not cause Z. Z is determined by its own mechanisms.')

    if graph_type == 'diamond':
        X, Y, Z, W = v['X'], v['Y'], v['Z'], v['W']
        if level == 'L1':
            qtext = _L1_OBSERVE_PHRASES[q_id % len(_L1_OBSERVE_PHRASES)](X) + f' {W}?'
            return _q(q_id, graph_type, level, story, qtext, {
                'A': f'{W} is likely higher (effect flows through both {Y} and {Z})',
                'B': f'{W} is unaffected because the two paths cancel out',
                'C': f'{W} is lower because of competing pathways',
                'D': f'Only {Y} is affected, not {W}',
            }, 'A',
                'In a diamond X->{Y,Z}->W, X=high implies Y=high and Z=high, both contributing to W=high.')
        if level == 'L2':
            l2_diamond = [
                f'If we intervene on {Y} alone (do({Y}=high)) without changing {X}, what happens to {W}?',
                f'Suppose we forcibly set {Y} to high while leaving {X} untouched. What is the effect on {W}?',
                f'If an experimenter sets do({Y}=high) but does not manipulate {X}, what happens to {W}?',
                f'Consider an intervention that makes {Y} high without affecting {X}. What about {W}?',
                f'We perform do({Y}=high) independently of {X}. How does {W} change?',
            ]
            return _q(q_id, graph_type, level, story, l2_diamond[q_id % len(l2_diamond)], {
                'A': f'{W} increases because {Y} directly causes {W}',
                'B': f'{W} is unaffected because we did not change {X}',
                'C': f'{W} increases fully as if {X} were high',
                'D': f'{Z} also changes, amplifying the effect on {W}',
            }, 'A',
                'do(Y=high) directly affects W through Y->W. Z is unchanged.')
        # L3
        qtext = _L3_CF_PHRASES[q_id % len(_L3_CF_PHRASES)](X, W)
        return _q(q_id, graph_type, level, story, qtext, {
            'A': f'{W} would have been higher (both pathways contribute positively)',
            'B': f'{W} would have stayed low (pathways cancel out)',
            'C': f'Only one pathway would activate, so {W} increases slightly',
            'D': 'Cannot determine without more information',
        }, 'A',
            'X=high would increase both Y and Z, both increasing W.')

    raise ValueError(f'Unknown graph type: {graph_type}')


def _q(q_id, graph_type, level, story, qtext, choices, answer, explanation):
    return {
        'id': q_id,
        'graph': graph_type,
        'level': level,
        'scenario': story,
        'question': qtext,
        'choices': choices,
        'answer': answer,
        'explanation': explanation,
    }


def shuffle_choices(question: Question, rng: random.Random) -> Question:
    """Permute A/B/C/D labels in-place so the correct letter is not predictable."""
    labels = ['A', 'B', 'C', 'D']
    perm = labels.copy()
    rng.shuffle(perm)
    mapping = dict(zip(labels, perm))
    new_choices = {mapping[k]: v for k, v in question['choices'].items()}
    question['choices'] = {k: new_choices[k] for k in labels}
    question['answer'] = mapping[question['answer']]
    return question


def question_fingerprint(q: Question) -> str:
    """Stable text-only hash to detect duplicate prompts."""
    body = q['scenario'] + '\n' + q['question'] + '\n' + '\n'.join(
        f"{k}:{q['choices'][k]}" for k in ['A', 'B', 'C', 'D']
    )
    return hashlib.md5(body.encode('utf-8')).hexdigest()


def build_benchmark(
    num_per_cell: int = NUM_QUESTIONS_PER_CELL,
    seed: int = SEED,
) -> list[Question]:
    """Generate the full benchmark: graph types x levels x num_per_cell questions.

    Choice labels are shuffled per question to avoid letter-frequency bias.
    Duplicate prompts (same fingerprint) are skipped where possible.
    """
    rng = random.Random(seed)
    benchmark: list[Question] = []
    seen: set[str] = set()
    q_id = 0
    max_tries = 50

    for graph_type, scenarios in SCENARIOS.items():
        for level in ['L1', 'L2', 'L3']:
            for _ in range(num_per_cell):
                tries = 0
                while True:
                    scenario = rng.choice(scenarios)
                    q = make_question(scenario, graph_type, level, q_id)
                    q_rng = random.Random(seed + 10_000 * q_id + 17)
                    q = shuffle_choices(q, q_rng)
                    fp = question_fingerprint(q)
                    if fp not in seen or tries >= max_tries:
                        seen.add(fp)
                        benchmark.append(q)
                        q_id += 1
                        break
                    tries += 1

    rng.shuffle(benchmark)
    return benchmark


def benchmark_summary(benchmark: Iterable[Question]) -> dict:
    """Return per-(graph,level) counts, answer distribution, and duplicate count."""
    benchmark = list(benchmark)
    counts = defaultdict(int)
    for q in benchmark:
        counts[(q['graph'], q['level'])] += 1
    answer_counts = Counter(q['answer'] for q in benchmark)
    fp_counts = Counter(question_fingerprint(q) for q in benchmark)
    duplicates = sum(c - 1 for c in fp_counts.values() if c > 1)
    return {
        'total': len(benchmark),
        'by_graph_level': dict(counts),
        'answers': dict(answer_counts),
        'duplicates': duplicates,
    }
