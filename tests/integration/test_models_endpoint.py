"""
Integration tests for GET /v1/models and GET /v1/models/{id}.
"""

import pytest


@pytest.mark.asyncio
async def test_list_models(async_client):
    """GET /v1/models returns a valid model list."""
    response = await async_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    first = data["data"][0]
    assert "id" in first
    assert "owned_by" in first


@pytest.mark.asyncio
async def test_list_models_includes_major_models(async_client):
    """Known models are present in the registry."""
    response = await async_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    model_ids = {m["id"] for m in data["data"]}
    assert "gpt-4o" in model_ids
    assert "gpt-4o-mini" in model_ids
    assert "claude-3-5-sonnet-20241022" in model_ids


@pytest.mark.asyncio
async def test_get_model_by_id(async_client):
    """GET /v1/models/{id} returns details for a specific model."""
    response = await async_client.get("/v1/models/gpt-4o")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "gpt-4o"
    assert data["owned_by"] == "openai"
    assert data["context_window"] == 128000
    assert data["capabilities"]["supports_tools"] is True


@pytest.mark.asyncio
async def test_get_unknown_model_returns_404(async_client):
    """GET /v1/models/unknown returns 404."""
    response = await async_client.get("/v1/models/unknown-fake-model-xyz")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_model_has_pricing(async_client):
    """Models with pricing have cost values."""
    response = await async_client.get("/v1/models/gpt-4o")
    assert response.status_code == 200
    data = response.json()
    assert data["input_cost_per_1m"] is not None
    assert data["output_cost_per_1m"] is not None
