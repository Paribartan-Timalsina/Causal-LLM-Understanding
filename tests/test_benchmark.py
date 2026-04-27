"""Smoke tests for benchmark generation and prompt formatting (no GPU/model required)."""

from collections import Counter

from causal_llm import (
    CAUSAL_GRAPHS,
    CONTENT_FREE_PROMPTS,
    PROMPT_FORMATTERS,
    PROMPTING_STRATEGIES,
    SCENARIOS,
    benchmark_summary,
    build_benchmark,
    make_question,
)
from causal_llm.benchmark import question_fingerprint


def test_benchmark_size_and_balance():
    bench = build_benchmark()
    assert len(bench) == 240, f'expected 240, got {len(bench)}'

    by_cell = Counter((q['graph'], q['level']) for q in bench)
    for graph in CAUSAL_GRAPHS:
        for level in ['L1', 'L2', 'L3']:
            assert by_cell[(graph, level)] == 20

    answers = Counter(q['answer'] for q in bench)
    assert set(answers) == {'A', 'B', 'C', 'D'}


def test_benchmark_is_deterministic():
    """Same seed must produce identical questions and identical answer distribution."""
    a = build_benchmark(seed=42)
    b = build_benchmark(seed=42)
    assert [q['id'] for q in a] == [q['id'] for q in b]
    assert [q['answer'] for q in a] == [q['answer'] for q in b]
    assert [question_fingerprint(q) for q in a] == [question_fingerprint(q) for q in b]


def test_make_question_all_combinations():
    for graph_type, scenarios in SCENARIOS.items():
        for level in ['L1', 'L2', 'L3']:
            q = make_question(scenarios[0], graph_type, level, q_id=0)
            assert q['graph'] == graph_type
            assert q['level'] == level
            assert q['answer'] in {'A', 'B', 'C', 'D'}
            assert set(q['choices']) == {'A', 'B', 'C', 'D'}


def test_no_duplicate_fingerprints():
    bench = build_benchmark()
    s = benchmark_summary(bench)
    assert s['duplicates'] == 0


def test_prompt_formatters_produce_strings():
    bench = build_benchmark()
    q = bench[0]
    for name, formatter in PROMPT_FORMATTERS.items():
        out = formatter(q)
        assert isinstance(out, str)
        assert out.rstrip().endswith('Answer:')
        assert q['scenario'] in out


def test_content_free_prompts():
    assert set(CONTENT_FREE_PROMPTS) == set(PROMPTING_STRATEGIES)
    for s, prompt in CONTENT_FREE_PROMPTS.items():
        assert 'N/A' in prompt
        assert prompt.rstrip().endswith('Answer:')
