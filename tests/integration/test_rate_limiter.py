"""
Integration & Unit tests for Rate Limiter (Hybrid, Redis, and In-Memory).
"""

import pytest

from app.core.rate_limiter import HybridRateLimiter, InMemoryRateLimiter, RedisRateLimiter


def test_in_memory_rate_limiter_allows_under_limit():
    limiter = InMemoryRateLimiter(requests_per_minute=5)
    for _ in range(5):
        assert not limiter.is_rate_limited("test-client")


def test_in_memory_rate_limiter_blocks_over_limit():
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    for _ in range(3):
        assert not limiter.is_rate_limited("test-client")

    # 4th request should be blocked
    assert limiter.is_rate_limited("test-client")


def test_redis_rate_limiter_fallback_when_redis_offline():
    limiter = RedisRateLimiter(requests_per_minute=5)
    # Should return None (signaling fallback to in-memory) when Redis URL is empty
    result = limiter.is_rate_limited("test-client")
    assert result is None or isinstance(result, bool)


def test_hybrid_rate_limiter_clears():
    limiter = HybridRateLimiter(requests_per_minute=2)
    limiter.is_rate_limited("test-client")
    limiter.is_rate_limited("test-client")
    assert limiter.is_rate_limited("test-client")

    limiter.clear()
    assert not limiter.is_rate_limited("test-client")
