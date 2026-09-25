"""
Integration tests for router configuration, explainability, and training endpoints.
"""

import pytest


@pytest.mark.asyncio
async def test_get_router_config(async_client):
    res = await async_client.get("/api/v1/router/config")
    assert res.status_code == 200
    data = res.json()
    assert "routing_table" in data
    assert "routing_mode" in data
    assert "confidence_threshold" in data


@pytest.mark.asyncio
async def test_update_router_config(async_client):
    res = await async_client.post(
        "/api/v1/router/config",
        json={"routing_mode": "shadow", "confidence_threshold": 0.75},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["routing_mode"] == "shadow"
    assert data["confidence_threshold"] == 0.75


@pytest.mark.asyncio
async def test_explain_router_endpoint(async_client):
    payload = {
        "messages": [{"role": "user", "content": "Build an async router in Python"}],
        "model": "gpt-4o",
    }
    res = await async_client.post("/api/v1/router/explain", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["requested_model"] == "gpt-4o"
    assert "rule_analysis" in data
    assert "ai_analysis" in data
    assert "shadow_mode" in data


@pytest.mark.asyncio
async def test_train_router_endpoint(async_client, db_session):
    """
    POST /api/v1/router/train must reach the real evaluate-then-hotswap pipeline
    (app.engine.router_trainer.train_router), not just accept a request. Two
    identically-pathed routes used to be registered (router_config.py and
    feedback.py) — FastAPI silently used whichever was added first, so this
    endpoint used to bypass the hot-swap safety check entirely. Seed enough
    labeled examples for a real (if small) training run, then confirm a
    training run with real accuracy metrics actually got recorded.
    """
    from app.db.models import RouterTrainingLabel

    seed_prompts = {
        "low": "What is the capital of France?",
        "medium": "Explain how neural networks work.",
        "high": "Implement a distributed rate limiter in Python with Redis and full test coverage.",
    }
    for label, prompt in seed_prompts.items():
        for _ in range(10):
            db_session.add(RouterTrainingLabel(
                prompt_snippet=prompt,
                model_requested="gpt-4o",
                complexity_label=label,
                label_source="human",
                confidence=1.0,
            ))
    db_session.commit()

    payload = {"force": True, "min_samples": 5}
    res = await async_client.post("/api/v1/router/train", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "training_queued"

    history_res = await async_client.get("/api/v1/router/training-history")
    assert history_res.status_code == 200
    runs = history_res.json()["runs"]
    assert len(runs) >= 1
    assert "new_model_accuracy" in runs[0]
    assert "model_swapped" in runs[0]
