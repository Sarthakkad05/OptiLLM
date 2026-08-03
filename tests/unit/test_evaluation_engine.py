"""
Unit tests for app/evaluation/ components — LLMJudge, HallucinationDetector, and Scorer.
"""

from app.evaluation.hallucination import HallucinationDetector, get_hallucination_detector
from app.evaluation.judge import LLMJudge, get_judge
from app.evaluation.scorer import categorize_efficiency, compute_efficiency_score


def test_llm_judge_scoring():
    judge = get_judge()
    messages = [{"role": "user", "content": "Write a Python function to compute fibonacci."}]
    response_text = "```python\ndef fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"

    res = judge.evaluate(messages, response_text)

    assert "quality_score" in res
    assert 0.0 <= res["quality_score"] <= 1.0
    assert "dimension_scores" in res
    dims = res["dimension_scores"]
    assert set(dims.keys()) == {"correctness", "relevance", "completeness", "conciseness", "safety"}
    assert dims["correctness"] > 0.8
    assert dims["safety"] == 1.0


def test_hallucination_detector_clean():
    detector = get_hallucination_detector()
    messages = [{"role": "user", "content": "What is Python?"}]
    response_text = "Python is a high-level programming language."

    res = detector.detect(messages, response_text)

    assert "hallucination_score" in res
    assert res["hallucination_score"] == 0.0
    assert res["risk_level"] == "low"


def test_hallucination_detector_unsupported_numbers():
    detector = get_hallucination_detector()
    messages = [{"role": "user", "content": "Summarize total revenue."}]
    reference_context = "The total revenue for 2025 was 100 million USD."
    response_text = "The revenue reached 999 million USD with 45 percent growth."

    res = detector.detect(messages, response_text, reference_context=reference_context)

    assert res["hallucination_score"] > 0.0
    assert len(res["unsupported_numbers"]) > 0
    assert "999" in res["unsupported_numbers"] or "45" in res["unsupported_numbers"]


def test_efficiency_scorer():
    eff_score = compute_efficiency_score(quality_score=0.90, cost_usd=0.00015)
    assert eff_score > 1000.0

    tier = categorize_efficiency(eff_score)
    assert tier in ["ultra_high", "high", "moderate", "low"]
