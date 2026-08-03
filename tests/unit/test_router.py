"""
Unit tests for app/engine/router.py — Phase 10 Router and Explainability
"""

from app.core.config import settings
from app.engine.router import (
    Complexity,
    explain_routing,
    get_routing_config,
    route,
    update_routing_config,
)


def test_route_simple_prompt():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    result = route(messages, requested_model="gpt-4o")

    assert "model_used" in result
    assert result["complexity"] in [Complexity.LOW, "low", "medium"]
    assert "routing_mode" in result


def test_route_complex_prompt():
    messages = [
        {
            "role": "user",
            "content": (
                "Write a Python script to perform async database migration with SQLAlchemy, "
                "handling deadlocks, transactions, and rollback strategies."
            ),
        }
    ]
    result = route(messages, requested_model="gpt-4o")

    assert "model_used" in result
    assert "score_breakdown" in result


def test_explain_routing():
    messages = [{"role": "user", "content": "Explain quantum computing in detail."}]
    report = explain_routing(messages, requested_model="gpt-4o")

    assert report["requested_model"] == "gpt-4o"
    assert "rule_analysis" in report
    assert "ai_analysis" in report
    assert "shadow_mode" in report
    assert "fallback" in report


def test_update_routing_config_mode_and_threshold():
    orig_mode = settings.ROUTING_MODE
    orig_thresh = settings.AI_ROUTER_CONFIDENCE_THRESHOLD

    try:
        updated = update_routing_config(
            routing_mode="ai",
            confidence_threshold=0.85,
        )
        assert updated["routing_mode"] == "ai"
        assert updated["confidence_threshold"] == 0.85
    finally:
        update_routing_config(
            routing_mode=orig_mode,
            confidence_threshold=orig_thresh,
        )
