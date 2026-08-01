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
