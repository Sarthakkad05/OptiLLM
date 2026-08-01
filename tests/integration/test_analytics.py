import pytest


@pytest.mark.asyncio
async def test_analytics_endpoint(async_client):
    response = await async_client.get("/api/v1/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "cost_over_time" in data
    assert "model_distribution" in data
    assert "recent_requests" in data


@pytest.mark.asyncio
async def test_analytics_latency_endpoint(async_client):
    response = await async_client.get("/api/v1/analytics/latency")
    assert response.status_code == 200
    data = response.json()
    assert "p50_ms" in data
    assert "p95_ms" in data
    assert "p99_ms" in data
    assert "per_provider" in data


@pytest.mark.asyncio
async def test_analytics_tokens_endpoint(async_client):
    response = await async_client.get("/api/v1/analytics/tokens")
    assert response.status_code == 200
    data = response.json()
    assert "trends" in data


@pytest.mark.asyncio
async def test_analytics_providers_endpoint(async_client):
    response = await async_client.get("/api/v1/analytics/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data


@pytest.mark.asyncio
async def test_analytics_savings_endpoint(async_client):
    response = await async_client.get("/api/v1/analytics/savings")
    assert response.status_code == 200
    data = response.json()
    assert "cache_savings_usd" in data
    assert "compression_savings_usd" in data
    assert "routing_savings_usd" in data
    assert "total_savings_usd" in data


@pytest.mark.asyncio
async def test_chat_completions_with_tag_header(async_client):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tagging test request"}],
    }
    headers = {"x-optillm-tag": "billing-team-alpha"}
    response = await async_client.post(
        "/v1/chat/completions", json=payload, headers=headers
    )
    assert response.status_code == 200

    # Query savings with tag filter
    res = await async_client.get("/api/v1/analytics/savings?tag=billing-team-alpha")
    assert res.status_code == 200
