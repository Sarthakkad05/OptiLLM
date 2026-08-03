"""
Integration tests for /metrics endpoint.
"""

import pytest


@pytest.mark.asyncio
async def test_get_metrics_endpoint(async_client):
    res = await async_client.get("/metrics")
    assert res.status_code == 200
    assert "optillm_http_requests_total" in res.text
