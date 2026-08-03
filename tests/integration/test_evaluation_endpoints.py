"""
Integration tests for Phase 11 evaluation API endpoints.
"""

import pytest


@pytest.mark.asyncio
async def test_evaluate_endpoint(async_client):
    payload = {
        "messages": [{"role": "user", "content": "Explain machine learning in simple terms."}],
        "response_text": "Machine learning is a field of AI where algorithms learn patterns from data.",
        "model_used": "gpt-4o",
        "cost_usd": 0.0005,
    }
    res = await async_client.post("/api/v1/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "quality_score" in data
    assert "dimension_scores" in data
    assert "hallucination_score" in data
    assert "efficiency_score" in data
    assert "efficiency_rating" in data


@pytest.mark.asyncio
async def test_ab_compare_endpoint(async_client):
    payload = {
        "messages": [{"role": "user", "content": "What is Python?"}],
        "model_a": "gpt-4o",
        "response_a": "Python is a high-level programming language used for web dev, AI, and automation.",
        "cost_a": 0.002,
        "model_b": "gemini-2.0-flash",
        "response_b": "Python is a popular programming language.",
        "cost_b": 0.0001,
    }
    res = await async_client.post("/api/v1/evaluate/compare", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["model_a"] == "gpt-4o"
    assert data["model_b"] == "gemini-2.0-flash"
    assert "winner" in data
    assert "winner_model" in data
    assert "efficiency_ratio" in data
    assert "recommendation_reason" in data


@pytest.mark.asyncio
async def test_quality_analytics_endpoint(async_client):
    res = await async_client.get("/api/v1/analytics/quality")
    assert res.status_code == 200
    data = res.json()

    assert "total_evaluated_requests" in data
    assert "avg_quality_score" in data
    assert "avg_hallucination_score" in data
    assert "dimension_averages" in data
    assert "model_quality_breakdown" in data
