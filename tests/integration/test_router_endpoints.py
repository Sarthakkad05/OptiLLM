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
async def test_train_router_endpoint(async_client):
    payload = {"use_synthetic_fallback": True, "max_logs": 100}
    res = await async_client.post("/api/v1/router/train", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "samples_trained" in data
    assert "feature_importances" in data
