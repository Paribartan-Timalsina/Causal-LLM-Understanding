"""Multiple-choice causal reasoning questions across Pearl's three levels."""

import hashlib
import random
from collections import Counter, defaultdict

from .config import NUM_QUESTIONS_PER_CELL, SEED
from .scenarios import SCENARIOS


_L1_OBSERVE = [
    lambda v: f'We observe that {v} is high. What do we expect for',
    lambda v: f'Suppose we notice that {v} is at a high level. What should we predict for',
    lambda v: f'Given that {v} is observed to be high, what is the likely outcome for',
    lambda v: f'If we see that {v} is high, what can we infer about',
    lambda v: f'Imagine we measure {v} and find it is high. What follows for',
]

_L2_INTERVENE = [
    lambda v: f'If we intervene and force {v} to be high (do({v}=high)), what happens to',
    lambda v: f'Suppose we externally set {v} to a high value. What is the effect on',
    lambda v: f'If we perform an intervention do({v}=high), what change do we expect in',
    lambda v: f'Consider forcibly making {v} high, breaking its natural causes. What happens to',
    lambda v: f'If an experimenter manipulates {v} to be high, what is the causal impact on',
]

_L3_CF = [
    lambda x, z: f'We observed {x}=low and {z}=low. If {x} had been high instead, what would {z} have been?',
    lambda x, z: f'Given that {x} was low and {z} was low, suppose {x} had been high. What would {z} be?',
    lambda x, z: f'In a world where {x}=low led to {z}=low, what would {z} have been if {x} were high?',
    lambda x, z: f'Counterfactual: we saw {x}=low, {z}=low. Had {x} been high, what about {z}?',
    lambda x, z: f'{x} was low and {z} ended up low. If we could go back and set {x} to high, what would {z} be?',
]

_PREFIXES = [
    '',
    'Consider this situation carefully. ',
    'Think about the causal relationships here. ',
    'Pay attention to the direction of causation. ',
    'Analyze the following causal scenario. ',
]


def make_question(scenario, graph_type, level, q_id):
    """Build one question dict (with the ground-truth answer letter)."""
    v = scenario['vars']
    story = _PREFIXES[q_id % len(_PREFIXES)] + scenario['story']

    if graph_type == 'chain':
        X, Y, Z = v['X'], v['Y'], v['Z']
        if level == 'L1':
            qtext = _L1_OBSERVE[q_id % len(_L1_OBSERVE)](X) + f' {Z}?'
            choices = {
                'A': f'{Z} is likely higher than average',
                'B': f'{Z} is likely lower than average',
                'C': f'{Z} is unaffected',
                'D': 'Cannot determine from the given information',
            }
            answer = 'A'
        elif level == 'L2':
            qtext = _L2_INTERVENE[q_id % len(_L2_INTERVENE)](X) + f' {Z}?'
            choices = {
                'A': f'{Z} increases (causal effect propagates through {Y})',
                'B': f'{Z} is unaffected because the intervention breaks the chain',
                'C': f'{Z} decreases due to the intervention',
                'D': 'The effect depends on unobserved confounders',
            }
            answer = 'A'
        else:
            qtext = _L3_CF[q_id % len(_L3_CF)](X, Z)
            choices = {
                'A': f'{Z} would have been higher',
                'B': f'{Z} would have remained low',
                'C': f'{Z} would have been lower',
                'D': 'Cannot determine without knowing the exact structural equations',
            }
            answer = 'A'

    elif graph_type == 'fork':
        C, X, Z = v['C'], v['X'], v['Z']
        if level == 'L1':
            qtext = _L1_OBSERVE[q_id % len(_L1_OBSERVE)](X) + f' {Z}?'
            choices = {
                'A': f'{Z} is likely higher than average',
                'B': f'{Z} is likely lower than average',
                'C': f'{Z} is completely unaffected by {X}',
                'D': f'{Z} is inversely related to {X}',
            }
            answer = 'A'
        elif level == 'L2':
            qtext = _L2_INTERVENE[q_id % len(_L2_INTERVENE)](X) + f' {Z}?'
            choices = {
                'A': f'{Z} increases because {X} and {Z} are correlated',
                'B': f'{Z} is unaffected because do({X}) severs the link from {C} to {X}',
                'C': f'{Z} decreases due to a compensatory mechanism',
                'D': f'{Z} doubles because of the intervention',
            }
            answer = 'B'
        else:
            qtext = _L3_CF[q_id % len(_L3_CF)](X, Z)
            choices = {
                'A': f'{Z} would have been high because {X} directly causes {Z}',
                'B': f'{Z} would still be low because {Z} is caused by {C} (which stays low), not {X}',
                'C': f'{Z} would have been average',
                'D': 'Cannot determine',
            }
            answer = 'B'

    elif graph_type == 'collider':
        X, Z, M = v['X'], v['Z'], v['M']
        if level == 'L1':
            l1_phrases = [
                f'In the general population (without conditioning on {M}), if we observe {X} is high, what do we expect for {Z}?',
                f'Without selecting on {M}, suppose we see {X} is high. What can we say about {Z}?',
                f'Ignoring {M} entirely, we notice {X} is high. Does that tell us anything about {Z}?',
                f'Before knowing anything about {M}, we observe {X} is high. What follows for {Z}?',
                f'If we simply observe {X}=high in the population (without conditioning on {M}), what about {Z}?',
            ]
            qtext = l1_phrases[q_id % len(l1_phrases)]
            choices = {
                'A': f'{Z} is likely higher',
                'B': f'{Z} is likely lower',
                'C': f'{Z} is unrelated to {X} (no information gained)',
                'D': f'{Z} depends on {M}',
            }
            answer = 'C'
        elif level == 'L2':
            l2_phrases = [
                f'We intervene and set do({X}=low). What is the *causal* effect of this intervention on {Z} (i.e., how does {Z} itself change as a result of the intervention)?',
                f'If we forcibly set {X} to low via do({X}=low), what is the causal effect on {Z}?',
                f'Suppose we perform do({X}=low). What is the causal impact on {Z} (independent of any conditioning on {M})?',
                f'What is the causal effect of intervening do({X}=low) on {Z}?',
                f'If an experimenter sets do({X}=low), what is the causal effect on {Z}?',
            ]
            qtext = l2_phrases[q_id % len(l2_phrases)]
            choices = {
                'A': f"{Z} increases because removing {X}'s contribution means {Z} must explain {M}",
                'B': f'{Z} is unaffected because {X} does not cause {Z}',
                'C': f'{Z} decreases proportionally',
                'D': 'The question is ill-defined',
            }
            answer = 'B'
        else:
            qtext = _L3_CF[q_id % len(_L3_CF)](X, Z)
            choices = {
                'A': f'{Z} would have been higher to compensate and keep {M} positive',
                'B': f'{Z} would have remained low (no causal connection)',
                'C': f'{Z} would have been lower',
                'D': f'{M} would have changed but {Z} stays the same',
            }
            answer = 'B'

    elif graph_type == 'diamond':
        X, Y, Z, W = v['X'], v['Y'], v['Z'], v['W']
        if level == 'L1':
            qtext = _L1_OBSERVE[q_id % len(_L1_OBSERVE)](X) + f' {W}?'
            choices = {
                'A': f'{W} is likely higher (effect flows through both {Y} and {Z})',
                'B': f'{W} is unaffected because the two paths cancel out',
                'C': f'{W} is lower because of competing pathways',
                'D': f'Only {Y} is affected, not {W}',
            }
            answer = 'A'
        elif level == 'L2':
            l2_diamond = [
                f'If we intervene on {Y} alone (do({Y}=high)) without changing {X}, what happens to {W}?',
                f'Suppose we forcibly set {Y} to high while leaving {X} untouched. What is the effect on {W}?',
                f'If an experimenter sets do({Y}=high) but does not manipulate {X}, what happens to {W}?',
                f'Consider an intervention that makes {Y} high without affecting {X}. What about {W}?',
                f'We perform do({Y}=high) independently of {X}. How does {W} change?',
            ]
            qtext = l2_diamond[q_id % len(l2_diamond)]
            choices = {
                'A': f'{W} increases because {Y} directly causes {W}',
                'B': f'{W} is unaffected because we did not change {X}',
                'C': f'{W} increases fully as if {X} were high',
                'D': f'{Z} also changes, amplifying the effect on {W}',
            }
            answer = 'A'
        else:
            qtext = _L3_CF[q_id % len(_L3_CF)](X, W)
            choices = {
                'A': f'{W} would have been higher (both pathways contribute positively)',
                'B': f'{W} would have stayed low (pathways cancel out)',
                'C': f'Only one pathway would activate, so {W} increases slightly',
                'D': 'Cannot determine without more information',
            }
            answer = 'A'

    else:
        raise ValueError(f'Unknown graph type: {graph_type}')

    return {
        'id': q_id,
        'graph': graph_type,
        'level': level,
        'scenario': story,
        'question': qtext,
        'choices': choices,
        'answer': answer,
    }


def shuffle_choices(question, rng):
    """Permute A/B/C/D so the correct letter isn't predictable from position."""
    labels = ['A', 'B', 'C', 'D']
    perm = labels.copy()
    rng.shuffle(perm)
    mapping = dict(zip(labels, perm))
    new_choices = {mapping[k]: v for k, v in question['choices'].items()}
    question['choices'] = {k: new_choices[k] for k in labels}
    question['answer'] = mapping[question['answer']]
    return question


def question_fingerprint(q):
    body = q['scenario'] + '\n' + q['question'] + '\n' + '\n'.join(
        f"{k}:{q['choices'][k]}" for k in ['A', 'B', 'C', 'D']
    )
    return hashlib.md5(body.encode('utf-8')).hexdigest()


def build_benchmark(num_per_cell=NUM_QUESTIONS_PER_CELL, seed=SEED):
    """Generate the full benchmark, shuffling choice labels per question.

    If a duplicate prompt comes up, retry up to 50 times with a different
    scenario before giving up and accepting the duplicate.
    """
    rng = random.Random(seed)
    benchmark = []
    seen = set()
    q_id = 0

    for graph_type, scenarios in SCENARIOS.items():
        for level in ['L1', 'L2', 'L3']:
            for _ in range(num_per_cell):
                q = None
                for _try in range(50):
                    scenario = rng.choice(scenarios)
                    q = make_question(scenario, graph_type, level, q_id)
                    q = shuffle_choices(q, random.Random(seed + 10_000 * q_id + 17))
                    fp = question_fingerprint(q)
                    if fp not in seen:
                        seen.add(fp)
                        break
                # else: accept the duplicate after 50 tries
                assert q is not None
                benchmark.append(q)
                q_id += 1

    rng.shuffle(benchmark)
    return benchmark


def benchmark_summary(benchmark):
    benchmark = list(benchmark)
    by_cell = defaultdict(int)
    for q in benchmark:
        by_cell[(q['graph'], q['level'])] += 1
    answers = Counter(q['answer'] for q in benchmark)
    fp_counts = Counter(question_fingerprint(q) for q in benchmark)
    duplicates = sum(c - 1 for c in fp_counts.values() if c > 1)
    return {
        'total': len(benchmark),
        'by_graph_level': dict(by_cell),
        'answers': dict(answers),
        'duplicates': duplicates,
    }
