import pytest

from app.engine.workflow_engine import (
    check_cache_node,
    classify_and_route_node,
    evaluate_quality_node,
    run_workflow_pipeline,
    should_retry_or_finish,
)


def test_workflow_node_functions(db_session):
    state = {
        "messages": [{"role": "user", "content": "What is Python?"}],
        "model_requested": "gpt-4o",
        "model_used": "gpt-4o",
        "messages_to_send": [{"role": "user", "content": "What is Python?"}],
        "cache_hit": False,
        "compressed": False,
        "routed": False,
        "tokens_input": 10,
        "tokens_output": 20,
        "tokens_saved": 0,
        "cost_usd": 0.001,
        "savings_usd": 0.0,
        "latency_ms": 100,
        "content": "Python is a programming language.",
        "quality_score": 0.95,
        "retry_count": 0,
        "db": db_session,
        "routing_reason": None,
    }

    res_cache = check_cache_node(state)
    assert res_cache.get("cache_hit") is False

    res_route = classify_and_route_node(state)
    assert "model_used" in res_route

    res_quality = evaluate_quality_node(state)
    assert res_quality["quality_score"] > 0.5

    decision = should_retry_or_finish(state)
    assert decision == "finish"


@pytest.mark.asyncio
async def test_run_workflow_pipeline(db_session):
    messages = [{"role": "user", "content": "Hello workflow engine"}]
    res = await run_workflow_pipeline(messages=messages, model="gpt-4o", db=db_session)

    assert "content" in res
    assert "quality_score" in res
    assert res["content"] != ""
