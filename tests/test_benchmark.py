"""Tests for benchmark generation."""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_llm.benchmark import generate_benchmark, make_question
from causal_llm.benchmark.scenarios import SCENARIOS
from causal_llm.benchmark.generator import _question_fingerprint


def test_make_question_chain_l1():
    scenario = SCENARIOS["chain"][0]
    q = make_question(scenario, "chain", "L1", 0)
    assert q["graph"] == "chain"
    assert q["level"] == "L1"
    assert q["answer"] in ["A", "B", "C", "D"]
    assert "scenario" in q
    assert "question" in q
    assert len(q["choices"]) == 4


def test_make_question_all_graph_levels():
    for graph_type, scenarios in SCENARIOS.items():
        for level in ["L1", "L2", "L3"]:
            q = make_question(scenarios[0], graph_type, level, 0)
            assert q["graph"] == graph_type
            assert q["level"] == level
            assert q["answer"] in ["A", "B", "C", "D"]


def test_generate_benchmark_default():
    benchmark = generate_benchmark(num_per_cell=5, seed=42)
    assert len(benchmark) == 5 * 5 * 3  # 5 graphs x 3 levels x 5 per cell

    graphs = Counter(q["graph"] for q in benchmark)
    levels = Counter(q["level"] for q in benchmark)

    assert len(graphs) == 5
    assert len(levels) == 3

    for g in ["chain", "fork", "collider", "diamond", "instrument"]:
        assert graphs[g] == 15  # 3 levels x 5

    for l in ["L1", "L2", "L3"]:
        assert levels[l] == 25  # 5 graphs x 5


def test_generate_benchmark_full():
    benchmark = generate_benchmark(num_per_cell=20, seed=42)
    assert len(benchmark) == 300

    fps = [_question_fingerprint(q) for q in benchmark]
    duplicates = len(fps) - len(set(fps))
    assert duplicates == 0, f"Found {duplicates} duplicate questions"


def test_benchmark_answer_distribution():
    benchmark = generate_benchmark(num_per_cell=20, seed=42)
    answers = Counter(q["answer"] for q in benchmark)
    for letter in ["A", "B", "C", "D"]:
        assert answers[letter] > 0, f"Missing answer label {letter}"


def test_benchmark_reproducibility():
    b1 = generate_benchmark(num_per_cell=5, seed=123)
    b2 = generate_benchmark(num_per_cell=5, seed=123)
    for q1, q2 in zip(b1, b2):
        assert q1["id"] == q2["id"]
        assert q1["question"] == q2["question"]
        assert q1["answer"] == q2["answer"]


if __name__ == "__main__":
    test_make_question_chain_l1()
    test_make_question_all_graph_levels()
    test_generate_benchmark_default()
    test_generate_benchmark_full()
    test_benchmark_answer_distribution()
    test_benchmark_reproducibility()
    print("All benchmark tests passed!")
