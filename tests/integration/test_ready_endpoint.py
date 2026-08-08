"""
Integration tests for GET /ready readiness probe.
"""

import pytest


@pytest.mark.asyncio
async def test_ready_endpoint_returns_200_or_503(async_client):
    """/ready returns a valid status (200 if configured, 503 if no providers)."""
    response = await async_client.get("/ready")
    assert response.status_code in (200, 503)
    data = response.json()
    # In test environment (no real API keys), either ready or not_ready
    assert "status" in data


@pytest.mark.asyncio
async def test_ready_endpoint_has_uptime(async_client):
    """Ready response includes uptime_seconds when healthy."""
    response = await async_client.get("/ready")
    if response.status_code == 200:
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_health_endpoint_has_version(async_client):
    """/health now returns version and uptime."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "uptime_seconds" in data
