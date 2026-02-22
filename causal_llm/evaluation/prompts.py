"""Prompt formatting strategies for causal reasoning evaluation."""

from ..benchmark.scenarios import FEW_SHOT_EXAMPLES


def format_zero_shot(question_dict):
    """Basic zero-shot prompt."""
    q = question_dict
    return f"""Consider the following causal scenario:

{q['scenario']}

Question: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Answer (respond with a single letter A/B/C/D):
Answer:"""


def format_few_shot(question_dict, num_examples=1):
    """Few-shot prompt with one demonstration example."""
    q = question_dict
    level = q["level"]
    ex = FEW_SHOT_EXAMPLES.get(level, FEW_SHOT_EXAMPLES["L1"])

    demo = f"""Example:
Scenario: {ex['scenario']}
Question: {ex['question']}
A) {ex['choices']['A']}  B) {ex['choices']['B']}  C) {ex['choices']['C']}  D) {ex['choices']['D']}
Answer: {ex['answer']}

"""
    return f"""{demo}Now answer this question:

Scenario: {q['scenario']}

Question: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Answer (respond with a single letter A/B/C/D):
Answer:"""


def format_chain_of_thought(question_dict):
    """Chain-of-thought prompt encouraging step-by-step reasoning."""
    q = question_dict
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


def format_causal_chain(question_dict):
    """Causal chain prompt: explicitly ask model to identify the graph first."""
    q = question_dict
    return f"""You are a causal reasoning expert. Analyze this scenario using Pearl's causal framework.

Scenario: {q['scenario']}

First, identify the causal graph structure:
- What are the variables?
- What are the directed causal edges?
- Is this a chain, fork, collider, diamond, or instrumental variable structure?

Then answer: {q['question']}

A) {q['choices']['A']}
B) {q['choices']['B']}
C) {q['choices']['C']}
D) {q['choices']['D']}

Answer (respond with a single letter A/B/C/D):
Answer:"""


PROMPT_FORMATTERS = {
    "zero_shot": format_zero_shot,
    "few_shot": format_few_shot,
    "chain_of_thought": format_chain_of_thought,
    "causal_chain": format_causal_chain,
}


def build_content_free_prompt(formatter):
    """Build a content-free prompt for PMI calibration."""
    dummy_q = {
        "scenario": "N/A",
        "question": "N/A",
        "choices": {"A": "N/A", "B": "N/A", "C": "N/A", "D": "N/A"},
        "answer": "A",
        "level": "L1",
        "id": "dummy",
        "graph": "chain",
    }
    return formatter(dummy_q)


CONTENT_FREE_PROMPTS = {
    strategy: build_content_free_prompt(fmt)
    for strategy, fmt in PROMPT_FORMATTERS.items()
}
