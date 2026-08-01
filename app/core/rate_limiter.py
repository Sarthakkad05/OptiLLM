"""
In-Memory Rate Limiter Middleware / Dependency.
Limits request frequency per minute per client key or client IP.
"""

import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException, Request, status

from app.core.config import settings


class InMemoryRateLimiter:
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


rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)


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
