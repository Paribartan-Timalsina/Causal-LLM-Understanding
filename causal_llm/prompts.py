"""Prompt formatters for each strategy + content-free prompts for PMI."""


def format_zero_shot(q):
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


def format_few_shot(q):
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


def format_chain_of_thought(q):
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


def format_causal_chain(q):
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


PROMPT_FORMATTERS = {
    'zero_shot': format_zero_shot,
    'few_shot': format_few_shot,
    'chain_of_thought': format_chain_of_thought,
    'causal_chain': format_causal_chain,
}


# Same templates with scenario/question/choices replaced by 'N/A'. Scoring
# the answer letters against this captures the model's letter prior, which
# we subtract from the real-prompt scores (PMI calibration).
_DUMMY_Q = {
    'scenario': 'N/A',
    'question': 'N/A',
    'choices': {'A': 'N/A', 'B': 'N/A', 'C': 'N/A', 'D': 'N/A'},
    'answer': 'A',
    'level': 'L1',
    'id': 'dummy',
    'graph': 'chain',
}
CONTENT_FREE_PROMPTS = {name: fn(_DUMMY_Q) for name, fn in PROMPT_FORMATTERS.items()}
