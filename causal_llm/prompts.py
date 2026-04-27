"""Prompt formatters for each strategy + content-free prompts for PMI calibration."""

from __future__ import annotations

from typing import Callable

Question = dict
Formatter = Callable[[Question], str]


def format_zero_shot(q: Question) -> str:
    return f"""Consider the following causal scenario:

{q['scenario']}

Question: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Answer (respond with a single letter A/B/C/D):
Answer:"""


_FEW_SHOT_EXAMPLES = {
    'L1': {
        'scenario': 'Sunlight (X) causes plants to grow taller (Y).',
        'question': 'We observe lots of sunlight. What do we expect for plant height?',
        'choices': {'A': 'Taller', 'B': 'Shorter', 'C': 'No change', 'D': 'Cannot tell'},
        'answer': 'A',
    },
    'L2': {
        'scenario': 'Weather (C) causes both ice cream sales (X) and drowning rates (Y). X does not cause Y.',
        'question': 'If we intervene and force high ice cream sales do(X=high), what happens to drowning rates?',
        'choices': {'A': 'Increases', 'B': 'No effect', 'C': 'Decreases', 'D': 'Doubles'},
        'answer': 'B',
    },
    'L3': {
        'scenario': 'Studying (X) causes good grades (Y). A student studied hard and got good grades.',
        'question': 'If the student had NOT studied, would they have gotten good grades?',
        'choices': {'A': 'Yes', 'B': 'No', 'C': 'Same grades', 'D': 'Better grades'},
        'answer': 'B',
    },
}


def format_few_shot(q: Question) -> str:
    ex = _FEW_SHOT_EXAMPLES.get(q['level'], _FEW_SHOT_EXAMPLES['L1'])
    demo = (
        f"Example:\n"
        f"Scenario: {ex['scenario']}\n"
        f"Question: {ex['question']}\n"
        f"A) {ex['choices']['A']}  B) {ex['choices']['B']}  "
        f"C) {ex['choices']['C']}  D) {ex['choices']['D']}\n"
        f"Answer: {ex['answer']}\n\n"
    )
    return f"""{demo}Now answer this question:

Scenario: {q['scenario']}

Question: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Answer (respond with a single letter A/B/C/D):
Answer:"""


def format_chain_of_thought(q: Question) -> str:
    return f"""Consider the following causal scenario:

{q['scenario']}

Question: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Let's think step by step about the causal relationships:
1. First, identify the causal structure (which variables cause which).
2. Then, determine what changes when we observe or intervene.
3. Finally, select the best answer.

Answer (respond with a single letter A/B/C/D):
Answer:"""


def format_causal_chain(q: Question) -> str:
    return f"""You are a causal reasoning expert. Analyze this scenario using Pearl's causal framework.

Scenario: {q['scenario']}

First, identify the causal graph structure:
- What are the variables?
- What are the directed causal edges?
- Is this a chain, fork, collider, or diamond structure?

Then answer: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Answer (respond with a single letter A/B/C/D):
Answer:"""


PROMPT_FORMATTERS: dict[str, Formatter] = {
    'zero_shot': format_zero_shot,
    'few_shot': format_few_shot,
    'chain_of_thought': format_chain_of_thought,
    'causal_chain': format_causal_chain,
}


def _build_content_free(formatter: Formatter) -> str:
    """Same template, but scenario/question/choices replaced with 'N/A'.

    Used for PMI calibration: scoring the answer letters against this prompt
    captures only the model's letter prior, so we can subtract it.
    """
    dummy = {
        'scenario': 'N/A',
        'question': 'N/A',
        'choices': {'A': 'N/A', 'B': 'N/A', 'C': 'N/A', 'D': 'N/A'},
        'answer': 'A',
        'level': 'L1',
        'id': 'dummy',
        'graph': 'chain',
    }
    return formatter(dummy)


CONTENT_FREE_PROMPTS: dict[str, str] = {
    name: _build_content_free(fn) for name, fn in PROMPT_FORMATTERS.items()
}
