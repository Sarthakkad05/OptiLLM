"""
Rate Limiter Middleware / Dependency.
Features:
  - TokenBucketRateLimiter: Allows bursts up to bucket_size and steady replenishment per second.
  - GlobalRateLimiter: Gateway-wide rate limit across all clients to prevent distributed overload.
  - AutoBlocklist: Automatically identifies and temporarily bans repeat 429 offenders.
  - HybridRateLimiter: Distributed Redis rate limiting with seamless in-memory fallback.
  - TPM (Tokens Per Minute) sliding window tracker.
  - Per-key custom overrides from APIKeyRecord (rpm_limit / tpm_limit).
  - Standard X-RateLimit-* response headers.
"""

import hashlib
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response, status

from app.core.config import settings
from app.engine.redis_cache import get_redis_client

logger = logging.getLogger("optillm.core.rate_limiter")


# ── Token Bucket Rate Limiter ─────────────────────────────────────────────────

class TokenBucketRateLimiter:
    """
    Token Bucket Rate Limiter.
    Allows bursts up to bucket capacity and continuously refills tokens at a fixed rate per second.
    Suitable for per-key or per-IP rate limiting with burst tolerance.
    """

    def __init__(self, requests_per_minute: int = 60, burst_factor: float = 1.5):
        self.requests_per_minute = requests_per_minute
        self.burst_factor = burst_factor
        self.default_capacity = max(1.0, float(requests_per_minute) * burst_factor)
        self.default_refill_rate = max(0.01, float(requests_per_minute) / 60.0)

        # key -> (current_tokens: float, last_update_time: float)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    def consume(
        self,
        key: str,
        tokens: float = 1.0,
        capacity: Optional[float] = None,
        refill_rate: Optional[float] = None,
    ) -> Tuple[bool, float, int]:
        """
        Attempts to consume `tokens` from the bucket for `key`.

        Returns:
            (allowed: bool, retry_after_seconds: float, remaining_tokens: int)
        """
        cap = capacity if capacity is not None else self.default_capacity
        rate = refill_rate if refill_rate is not None else self.default_refill_rate
        now = time.time()

        if key not in self._buckets:
            current_tokens = cap
            last_time = now
        else:
            current_tokens, last_time = self._buckets[key]
            # Replenish tokens based on elapsed time
            elapsed = max(0.0, now - last_time)
            current_tokens = min(cap, current_tokens + elapsed * rate)

        if current_tokens >= tokens:
            new_tokens = current_tokens - tokens
            self._buckets[key] = (new_tokens, now)
            return True, 0.0, int(new_tokens)
        else:
            # Insufficient tokens
            needed = tokens - current_tokens
            retry_after = needed / rate if rate > 0 else 60.0
            self._buckets[key] = (current_tokens, now)
            return False, retry_after, int(current_tokens)

    def remaining(
        self,
        key: str,
        capacity: Optional[float] = None,
        refill_rate: Optional[float] = None,
    ) -> int:
        """Returns current available token count for a key."""
        cap = capacity if capacity is not None else self.default_capacity
        rate = refill_rate if refill_rate is not None else self.default_refill_rate
        now = time.time()

        if key not in self._buckets:
            return int(cap)

        current_tokens, last_time = self._buckets[key]
        elapsed = max(0.0, now - last_time)
        return int(min(cap, current_tokens + elapsed * rate))

    def reset_at(
        self,
        key: str,
        capacity: Optional[float] = None,
        refill_rate: Optional[float] = None,
    ) -> int:
        """Returns unix timestamp when bucket will be completely full."""
        cap = capacity if capacity is not None else self.default_capacity
        rate = refill_rate if refill_rate is not None else self.default_refill_rate
        current = self.remaining(key, capacity=cap, refill_rate=rate)
        needed = max(0.0, cap - current)
        return int(time.time() + (needed / rate if rate > 0 else 60))

    def clear(self, key: Optional[str] = None):
        if key:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()


# ── Auto Blocklist for Repeat 429 Offenders ───────────────────────────────────

class AutoBlocklist:
    """
    Tracks repeat rate-limit violations and temporarily blocks hostile or
    abusive clients across consecutive 429 breaches.
    """

    def __init__(
        self,
        threshold: int = 5,
        window_seconds: float = 60.0,
        block_duration: float = 300.0,
    ):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.block_duration = block_duration
        self._violations: Dict[str, List[float]] = defaultdict(list)
        self._blocked_until: Dict[str, float] = {}

    def is_blocked(self, client_id: str) -> Tuple[bool, int]:
        """
        Checks if client is currently blocked.
        Returns (is_blocked: bool, remaining_block_seconds: int).
        """
        now = time.time()
        expiry = self._blocked_until.get(client_id, 0.0)
        if now < expiry:
            return True, int(expiry - now)
        elif client_id in self._blocked_until:
            del self._blocked_until[client_id]
        return False, 0

    def record_violation(self, client_id: str) -> bool:
        """
        Records a 429 violation for a client.
        If violations within window exceed threshold, blocks the client.
        Returns True if the client was newly placed on the blocklist.
        """
        now = time.time()
        cutoff = now - self.window_seconds
        # Clean older violation timestamps
        recent = [t for t in self._violations[client_id] if t > cutoff]
        recent.append(now)
        self._violations[client_id] = recent

        if len(recent) >= self.threshold:
            self._blocked_until[client_id] = now + self.block_duration
            logger.warning(
                "Security Alert: Client '%s' blocked for %ds due to %d rate limit violations.",
                client_id,
                int(self.block_duration),
                len(recent),
            )
            # Reset violation list once blocked
            self._violations[client_id] = []
            return True
        return False

    def unblock(self, client_id: str):
        self._blocked_until.pop(client_id, None)
        self._violations.pop(client_id, None)

    def clear(self):
        self._blocked_until.clear()
        self._violations.clear()


# ── In-Memory Sliding Window (Legacy / Direct) ────────────────────────────────

class _InMemorySlidingWindowRateLimiter:
    """Sliding window in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_rate_limited(self, key: str) -> bool:
        now = time.time()
        window_start = now - 60.0

        valid_timestamps = [t for t in self.requests[key] if t > window_start]
        self.requests[key] = valid_timestamps

        if len(valid_timestamps) >= self.requests_per_minute:
            return True

        self.requests[key].append(now)
        return False

    def remaining(self, key: str) -> int:
        now = time.time()
        window_start = now - 60.0
        valid = [t for t in self.requests.get(key, []) if t > window_start]
        return max(0, self.requests_per_minute - len(valid))

    def reset_at(self, key: str) -> int:
        timestamps = self.requests.get(key, [])
        if not timestamps:
            return int(time.time()) + 60
        return int(min(timestamps) + 60)

    def clear(self):
        self.requests.clear()


# ── TPM (Tokens Per Minute) Tracker ──────────────────────────────────────────

class _TPMTracker:
    """Tracks token consumption per API key in a sliding 60-second window."""

    def __init__(self):
        self._windows: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

    def _clean_window(self, key: str) -> List[Tuple[float, int]]:
        now = time.time()
        window_start = now - 60.0
        valid = [(ts, t) for ts, t in self._windows[key] if ts > window_start]
        self._windows[key] = valid
        return valid

    def record(self, key: str, tokens: int):
        self._windows[key].append((time.time(), tokens))

    def used_this_minute(self, key: str) -> int:
        valid = self._clean_window(key)
        return sum(t for _, t in valid)

    def is_tpm_limited(self, key: str, tpm_limit: int) -> bool:
        return self.used_this_minute(key) >= tpm_limit


# ── Redis Rate Limiter ────────────────────────────────────────────────────────

class RedisRateLimiter:
    """Distributed Redis rate limiter using fixed minute-window counter with TTL."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute

    def is_rate_limited(self, key: str) -> Optional[bool]:
        client = get_redis_client()
        if not client:
            return None

        try:
            current_minute = int(time.time() // 60)
            redis_key = f"ratelimit:{key}:{current_minute}"

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


# ── Hybrid Rate Limiter ───────────────────────────────────────────────────────

class HybridRateLimiter:
    """
    Hybrid Rate Limiter combining Redis distributed limiting with
    in-memory TokenBucket / SlidingWindow fallback.
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

    def remaining(self, key: str) -> int:
        return self.in_memory.remaining(key)

    def reset_at(self, key: str) -> int:
        return self.in_memory.reset_at(key)

    def clear(self):
        self.in_memory.clear()


# Backward-compatible aliases
InMemoryRateLimiter = _InMemorySlidingWindowRateLimiter

rate_limiter = HybridRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)
token_bucket_limiter = TokenBucketRateLimiter(
    requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    burst_factor=settings.RATE_LIMIT_BURST_FACTOR,
)
global_rate_limiter = TokenBucketRateLimiter(
    requests_per_minute=settings.GLOBAL_RATE_LIMIT_PER_MINUTE,
    burst_factor=1.2,
)
auto_blocklist = AutoBlocklist(
    threshold=settings.AUTO_BLOCKLIST_THRESHOLD,
    window_seconds=60.0,
    block_duration=float(settings.AUTO_BLOCKLIST_DURATION_SECONDS),
)
tpm_tracker = _TPMTracker()


def record_token_usage(api_key: str, tokens: int):
    tpm_tracker.record(api_key, tokens)


def get_key_limits(api_key: str) -> Tuple[int, int]:
    return settings.RATE_LIMIT_PER_MINUTE, settings.TPM_LIMIT_PER_KEY


def get_rate_limit_headers(api_key: str, rpm_limit: int, tpm_limit: int) -> Dict[str, str]:
    remaining_rpm = token_bucket_limiter.remaining(api_key)
    reset_ts = token_bucket_limiter.reset_at(api_key)
    used_tpm = tpm_tracker.used_this_minute(api_key)

    return {
        "X-RateLimit-Limit-Requests": str(rpm_limit),
        "X-RateLimit-Remaining-Requests": str(remaining_rpm),
        "X-RateLimit-Reset-Requests": str(reset_ts),
        "X-RateLimit-Limit-Tokens": str(tpm_limit),
        "X-RateLimit-Remaining-Tokens": str(max(0, tpm_limit - used_tpm)),
    }


async def check_rate_limit(request: Request):
    """
    FastAPI dependency enforcing:
      1. AutoBlocklist: Immediate rejection of repeatedly abusive clients.
      2. Global Rate Limit: Protects gateway from overall distributed multi-IP denial-of-service.
      3. Per-Client Token Bucket (RPM + Burst tolerance): Allows bursts up to capacity.
      4. Per-Client TPM Tracker: Enforces token throughput boundaries.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        client_key = auth_header.split(" ", 1)[1].strip()
    else:
        client_key = request.client.host if request.client else "127.0.0.1"

    # 1. Check Auto-Blocklist
    if settings.AUTO_BLOCKLIST_ENABLED:
        blocked, rem_sec = auto_blocklist.is_blocked(client_key)
        if blocked:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Client temporarily blocked due to repeated rate limit violations. Try again in {rem_sec}s.",
                headers={
                    "Retry-After": str(rem_sec),
                    "X-RateLimit-Remaining-Requests": "0",
                },
            )

    # 2. Check Global Rate Limit (Gateway-wide)
    global_allowed, global_retry, _ = global_rate_limiter.consume("gateway_global", tokens=1.0)
    if not global_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Gateway global rate limit exceeded. Retry after {int(global_retry)}s.",
            headers={"Retry-After": str(int(global_retry))},
        )

    # 3. Resolve per-key limits from DB
    rpm_limit = settings.RATE_LIMIT_PER_MINUTE
    tpm_limit = settings.TPM_LIMIT_PER_KEY

    try:
        from app.api.endpoints.keys import APIKeyRecord
        from app.db.session import SessionLocal

        key_hash = hashlib.sha256(client_key.encode()).hexdigest()
        with SessionLocal() as db:
            record = db.query(APIKeyRecord).filter(
                APIKeyRecord.key_hash == key_hash,
                APIKeyRecord.is_active == True,
            ).first()
            if record:
                if getattr(record, "rpm_limit", None):
                    rpm_limit = record.rpm_limit
                if getattr(record, "tpm_limit", None):
                    tpm_limit = record.tpm_limit
    except Exception:
        pass  # Fail open to global defaults

    # 4. Check Per-Client Rate Limit (Token Bucket)
    burst_cap = max(5.0, float(rpm_limit) * settings.RATE_LIMIT_BURST_FACTOR)
    refill = max(0.01, float(rpm_limit) / 60.0)

    # Check hybrid Redis first if configured
    redis_limited = rate_limiter.redis_limiter.is_rate_limited(client_key)
    if redis_limited is True:
        auto_blocklist.record_violation(client_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {rpm_limit} requests/minute.",
            headers={"Retry-After": "30", "X-RateLimit-Limit-Requests": str(rpm_limit)},
        )

    # Check Token Bucket
    allowed, retry_after, rem_tokens = token_bucket_limiter.consume(
        client_key, tokens=1.0, capacity=burst_cap, refill_rate=refill
    )
    if not allowed:
        auto_blocklist.record_violation(client_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {rpm_limit} requests/minute. "
                f"Retry after {max(1, int(retry_after))}s."
            ),
            headers={
                "Retry-After": str(max(1, int(retry_after))),
                "X-RateLimit-Limit-Requests": str(rpm_limit),
                "X-RateLimit-Remaining-Requests": "0",
            },
        )

    # 5. Check TPM Limit
    if tpm_tracker.is_tpm_limited(client_key, tpm_limit):
        auto_blocklist.record_violation(client_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Token rate limit exceeded: {tpm_limit} tokens/minute. Retry in a moment.",
            headers={
                "X-RateLimit-Limit-Tokens": str(tpm_limit),
                "X-RateLimit-Remaining-Tokens": "0",
            },
        )
