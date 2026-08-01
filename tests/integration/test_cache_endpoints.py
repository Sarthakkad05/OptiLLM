import pytest


@pytest.mark.asyncio
async def test_cache_info_endpoint(async_client):
    response = await async_client.get("/api/v1/cache/info")
    assert response.status_code == 200
    data = response.json()

    assert "total_cached_responses" in data
    assert "redis_available" in data
    assert "cache_namespace" in data
    assert "cache_ttl_seconds" in data


@pytest.mark.asyncio
async def test_cache_warm_endpoint(async_client):
    response = await async_client.post("/api/v1/cache/warm")
    assert response.status_code == 200
    data = response.json()

    assert "warmed_entries" in data
    assert "message" in data


@pytest.mark.asyncio
async def test_cache_clear_endpoint(async_client):
    response = await async_client.delete("/api/v1/cache/clear")
    assert response.status_code == 200
    data = response.json()

    assert "deleted_entries" in data
    assert "message" in data
