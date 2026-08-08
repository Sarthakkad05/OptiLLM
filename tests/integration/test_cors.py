"""
Integration tests for CORS middleware configuration.
"""

import pytest


@pytest.mark.asyncio
async def test_cors_options_request(async_client):
    """OPTIONS request returns CORS headers."""
    response = await async_client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
