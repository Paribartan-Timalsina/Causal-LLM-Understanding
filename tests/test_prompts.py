"""Tests for prompt formatting strategies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_llm.evaluation.prompts import (
    format_zero_shot,
    format_few_shot,
    format_chain_of_thought,
    format_causal_chain,
    PROMPT_FORMATTERS,
    build_content_free_prompt,
)


SAMPLE_QUESTION = {
    "id": 0,
    "graph": "chain",
    "level": "L1",
    "scenario": "A causes B causes C.",
    "question": "If A is high, what happens to C?",
    "choices": {
        "A": "C increases",
        "B": "C decreases",
        "C": "C is unaffected",
        "D": "Cannot determine",
    },
    "answer": "A",
    "explanation": "Chain propagation.",
}


def test_format_zero_shot():
    prompt = format_zero_shot(SAMPLE_QUESTION)
    assert isinstance(prompt, str)
    assert "A causes B causes C" in prompt
    assert "Answer:" in prompt
    assert "A)" in prompt
    assert "B)" in prompt


def test_format_few_shot():
    prompt = format_few_shot(SAMPLE_QUESTION)
    assert isinstance(prompt, str)
    assert "Example:" in prompt
    assert "Now answer" in prompt
    assert "Answer:" in prompt


def test_format_chain_of_thought():
    prompt = format_chain_of_thought(SAMPLE_QUESTION)
    assert isinstance(prompt, str)
    assert "step by step" in prompt.lower()
    assert "Answer:" in prompt


def test_format_causal_chain():
    prompt = format_causal_chain(SAMPLE_QUESTION)
    assert isinstance(prompt, str)
    assert "causal" in prompt.lower()
    assert "Answer:" in prompt


def test_all_formatters_in_registry():
    expected = {"zero_shot", "few_shot", "chain_of_thought", "causal_chain"}
    assert set(PROMPT_FORMATTERS.keys()) == expected


def test_content_free_prompts():
    for name, formatter in PROMPT_FORMATTERS.items():
        prompt = build_content_free_prompt(formatter)
        assert isinstance(prompt, str)
        assert "N/A" in prompt
        assert "Answer:" in prompt


if __name__ == "__main__":
    test_format_zero_shot()
    test_format_few_shot()
    test_format_chain_of_thought()
    test_format_causal_chain()
    test_all_formatters_in_registry()
    test_content_free_prompts()
    print("All prompt tests passed!")
