"""Tests for scoring utilities (structural/mock-based)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_llm.evaluation.prompts import CONTENT_FREE_PROMPTS


def test_content_free_prompts_exist():
    expected = {"zero_shot", "few_shot", "chain_of_thought", "causal_chain"}
    assert set(CONTENT_FREE_PROMPTS.keys()) == expected


def test_content_free_prompts_not_empty():
    for strategy, prompt in CONTENT_FREE_PROMPTS.items():
        assert len(prompt) > 50, f"Content-free prompt for {strategy} is too short"


def test_content_free_prompts_contain_na():
    for strategy, prompt in CONTENT_FREE_PROMPTS.items():
        assert "N/A" in prompt, f"Content-free prompt for {strategy} missing N/A"


def test_content_free_prompts_end_with_answer():
    for strategy, prompt in CONTENT_FREE_PROMPTS.items():
        assert prompt.rstrip().endswith("Answer:"), (
            f"Content-free prompt for {strategy} should end with 'Answer:'"
        )


if __name__ == "__main__":
    test_content_free_prompts_exist()
    test_content_free_prompts_not_empty()
    test_content_free_prompts_contain_na()
    test_content_free_prompts_end_with_answer()
    print("All scoring tests passed!")
