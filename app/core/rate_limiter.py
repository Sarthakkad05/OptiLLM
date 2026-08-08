"""
Rate Limiter Middleware / Dependency.
Supports Redis-backed distributed rate limiting when REDIS_URL is configured,
with graceful fallback to in-memory sliding window rate limiting when Redis is offline or unconfigured.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import HTTPException, Request, status

from app.core.config import settings
from app.engine.redis_cache import get_redis_client

logger = logging.getLogger("optillm.core.rate_limiter")


class _InMemorySlidingWindowRateLimiter:
    """
    Sliding window in-memory rate limiter.
    Stores timestamps of requests per client key or IP.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_rate_limited(self, key: str) -> bool:
        now = time.time()
        window_start = now - 60.0

        # Filter timestamps within the last 60 seconds
        valid_timestamps = [t for t in self.requests[key] if t > window_start]
        self.requests[key] = valid_timestamps

        if len(valid_timestamps) >= self.requests_per_minute:
            return True

        self.requests[key].append(now)
        return False

    def clear(self):
        self.requests.clear()


class RedisRateLimiter:
    """
    Distributed Redis rate limiter using fixed minute-window counter with TTL.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute

    def is_rate_limited(self, key: str) -> Optional[bool]:
        client = get_redis_client()
        if not client:
            return None  # Signal caller to fallback to in-memory

        try:
            current_minute = int(time.time() // 60)
            redis_key = f"ratelimit:{key}:{current_minute}"

            # Pipeline INCR and EXPIRE for atomicity
            pipe = client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, 65)
            results = pipe.execute()

            request_count = results[0]
            return request_count > self.requests_per_minute
        except Exception as exc:
            logger.warning(
                "Redis rate limiter error (%s) — falling back to in-memory.", exc
            )
            return None


class HybridRateLimiter:
    """
    Hybrid Rate Limiter that attempts Redis distributed rate limiting first,
    falling back to in-memory rate limiting if Redis is unavailable.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.in_memory = _InMemorySlidingWindowRateLimiter(requests_per_minute=requests_per_minute)
        self.redis_limiter = RedisRateLimiter(requests_per_minute=requests_per_minute)

    def is_rate_limited(self, key: str) -> bool:
        redis_result = self.redis_limiter.is_rate_limited(key)
        if redis_result is not None:
            return redis_result
        return self.in_memory.is_rate_limited(key)

    def clear(self):
        self.in_memory.clear()


# Backward-compatible aliases
InMemoryRateLimiter = _InMemorySlidingWindowRateLimiter

rate_limiter = HybridRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)


async def check_rate_limit(request: Request):
    """
    FastAPI dependency enforcing per-client rate limits.
    Determines client key by Bearer token or client IP.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        client_key = auth_header.split(" ", 1)[1].strip()
    else:
        client_key = request.client.host if request.client else "127.0.0.1"

    if rate_limiter.is_rate_limited(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute.",
        )
