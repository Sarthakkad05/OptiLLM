"""
Integration tests for Graceful Degradation and Protection Limits (Phase 3).

Tests:
1. Request size limit middleware (413 Payload Too Large)
2. Message count limit (413 Payload Too Large)
3. Prompt length limit (413 Payload Too Large)
4. FAISS-only cache fallback when Redis is down
5. 503 Service Unavailable with Retry-After when all providers circuit-break
6. Structured health /ready and /health/ready reporting
7. Async DB session dependency execution
"""

import pytest
from sqlalchemy import text
from unittest.mock import patch, MagicMock

from app.core.circuit_breaker import circuit_breaker
from app.core.config import settings
from app.db.session import get_async_db
from app.engine.cache import check_cache, insert_cache
from app.providers.registry import provider_registry


@pytest.mark.asyncio
async def test_request_body_size_limit_rejected(async_client):
    """Requests with Content-Length exceeding MAX_REQUEST_BYTES are rejected with 413."""
    oversized_length = settings.MAX_REQUEST_BYTES + 1024
    response = await async_client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer sk-optillm-dev-key",
            "Content-Length": str(oversized_length),
            "Content-Type": "application/json",
        },
        content=b'{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}',
    )
    assert response.status_code == 413
    data = response.json()
    assert "Payload too large" in data.get("detail", "")


@pytest.mark.asyncio
async def test_message_count_limit_rejected(async_client):
    """Requests exceeding MAX_MESSAGES_PER_REQUEST are rejected with 413."""
    excessive_messages = [
        {"role": "user", "content": f"msg {i}"}
        for i in range(settings.MAX_MESSAGES_PER_REQUEST + 5)
    ]
    response = await async_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-optillm-dev-key"},
        json={
            "model": "gpt-4o",
            "messages": excessive_messages,
        },
    )
    assert response.status_code == 413
    data = response.json()
    assert "Too many messages" in data.get("detail", "")


@pytest.mark.asyncio
async def test_prompt_length_limit_completions(async_client):
    """Legacy completions endpoint rejects prompts exceeding MAX_PROMPT_LENGTH."""
    oversized_prompt = "A" * (settings.MAX_PROMPT_LENGTH + 10)
    response = await async_client.post(
        "/v1/completions",
        headers={"Authorization": "Bearer sk-optillm-dev-key"},
        json={
            "model": "gpt-4o",
            "prompt": oversized_prompt,
        },
    )
    assert response.status_code == 413
    data = response.json()
    assert "Prompt too long" in data.get("detail", "")


@pytest.mark.asyncio
async def test_redis_down_faiss_fallback(db_session):
    """When Redis is unavailable, cache insertion and lookup succeed via local FAISS and DB."""
    test_messages = [{"role": "user", "content": "What is graceful degradation in distributed systems?"}]
    response_content = "Graceful degradation allows systems to maintain partial function during failures."

    from app.engine.faiss_store import rebuild_from_entries
    rebuild_from_entries([])

    # Simulate Redis being down
    with patch("app.engine.cache.lookup_redis_cache", return_value=None), \
         patch("app.engine.cache.insert_redis_cache", return_value=None):

        # Insert into cache (writes to FAISS + DB, skips Redis cleanly)
        insert_cache(
            messages=test_messages,
            response_text=response_content,
            model="gpt-4o",
            tokens_input=10,
            tokens_output=15,
            db=db_session,
        )

        # Lookup should succeed via local FAISS
        match = check_cache(
            messages=test_messages,
            similarity_threshold=0.85,
            db=db_session,
        )

        assert match is not None
        assert match["response_text"] == response_content
        assert match["tokens_input"] == 10
        assert match["tokens_output"] == 15


@pytest.mark.asyncio
async def test_all_providers_circuit_broken_returns_503(async_client, monkeypatch):
    """When in production and all providers are circuit-broken or failed, return HTTP 503 with Retry-After."""
    monkeypatch.setattr(settings, "APP_ENV", "production")

    # Mock provider available list to pretend openai is configured
    monkeypatch.setattr(provider_registry, "list_available_providers", lambda: ["openai"])

    # Trip circuit breaker for openai
    for _ in range(settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD + 2):
        circuit_breaker.record_failure("openai")

    # Make load balancer return None or unavailable
    response = await async_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-optillm-dev-key"},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, is any provider up?"}],
            "optillm": {"bypass_cache": True},
        },
    )

    assert response.status_code == 503
    assert response.headers.get("retry-after") == "30"
    data = response.json()
    assert "unavailable" in data.get("detail", "").lower() or "circuit-broken" in data.get("detail", "").lower()


@pytest.mark.asyncio
async def test_structured_health_ready_subsystems(async_client):
    """GET /health/ready returns detailed status for all subsystems."""
    response = await async_client.get("/health/ready")
    assert response.status_code in (200, 503)
    data = response.json()

    assert "status" in data
    assert data["status"] in ("ready", "degraded", "down")
    assert "service" in data
    assert "version" in data
    assert "uptime_seconds" in data
    assert "checks" in data

    checks = data["checks"]
    assert "database" in checks
    assert "redis" in checks
    assert "faiss" in checks
    assert "embedding_model" in checks
    assert "providers" in checks


@pytest.mark.asyncio
async def test_async_db_dependency():
    """Verify get_async_db yields an active session and executes queries."""
    generator = get_async_db()
    session = await anext(generator)
    try:
        assert session is not None
        # Execute test query
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
    finally:
        try:
            await anext(generator)
        except StopAsyncIteration:
            pass
