import pytest


@pytest.mark.asyncio
async def test_providers_status_endpoint(async_client):
    response = await async_client.get("/api/v1/providers/status")
    assert response.status_code == 200
    data = response.json()

    assert "routing_strategy" in data
    assert "available_providers_count" in data
    assert "providers" in data
    assert isinstance(data["providers"], list)
    assert len(data["providers"]) > 0

    provider_names = [p["name"] for p in data["providers"]]
    assert "openai" in provider_names
    assert "gemini" in provider_names
    assert "anthropic" in provider_names
