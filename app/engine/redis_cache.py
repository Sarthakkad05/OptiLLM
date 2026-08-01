"""
Distributed Redis Semantic Cache Engine.
Provides horizontally scalable vector similarity search, TTL expiration,
and multi-tenant namespace isolation backed by Redis.

Falls back gracefully to in-process FAISS if Redis is offline or unconfigured.
"""

import json
import logging
from typing import Any, Dict, Optional

import numpy as np

from app.core.config import settings

logger = logging.getLogger("optillm.engine.redis_cache")

_redis_client = None


def get_redis_client():
    """Lazily initializes and returns Redis connection client if REDIS_URL is configured."""
    global _redis_client
    if not settings.REDIS_URL:
        return None

    if _redis_client is None:
        try:
            import redis

            _redis_client = redis.Redis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_timeout=1.5
            )
            _redis_client.ping()
            logger.info(
                "Connected to Redis distributed cache at %s", settings.REDIS_URL
            )
        except Exception as exc:
            logger.warning(
                "Failed to connect to Redis (%s) — using FAISS fallback.", exc
            )
            _redis_client = None

    return _redis_client


def is_redis_available() -> bool:
    """Returns True if Redis client is connected and reachable."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        return client.ping()
    except Exception:
        return False


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two 1D float vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def lookup_redis_cache(
    query_embedding: np.ndarray,
    threshold: Optional[float] = None,
    namespace: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Searches Redis for a cached vector embedding exceeding similarity threshold.
    """
    client = get_redis_client()
    if not client:
        return None

    ns = namespace or settings.CACHE_NAMESPACE
    sim_threshold = (
        threshold if threshold is not None else settings.CACHE_SIMILARITY_THRESHOLD
    )
    pattern = f"optillm:{ns}:cache:*"

    try:
        keys = client.keys(pattern)
        if not keys:
            return None

        best_match = None
        best_score = -1.0

        for key in keys:
            raw_data = client.get(key)
            if not raw_data:
                continue

            entry = json.loads(raw_data)
            cached_embedding = np.array(entry["embedding"], dtype="float32")

            score = _cosine_similarity(query_embedding, cached_embedding)
            if score >= sim_threshold and score > best_score:
                best_score = score
                best_match = {
                    "response_text": entry["response_text"],
                    "model": entry["model"],
                    "tokens_input": entry["tokens_input"],
                    "tokens_output": entry["tokens_output"],
                    "similarity": round(score, 4),
                }

        if best_match:
            logger.info("Redis CACHE HIT | similarity=%.4f | ns=%s", best_score, ns)
            return best_match

    except Exception as exc:
        logger.warning("Redis lookup error (%s) — degrading to FAISS.", exc)

    return None


def insert_redis_cache(
    entry_id: str,
    prompt_text: str,
    query_embedding: np.ndarray,
    response_text: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    namespace: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """
    Inserts entry into Redis cache with vector embedding, response metadata, and TTL.
    """
    client = get_redis_client()
    if not client:
        return False

    ns = namespace or settings.CACHE_NAMESPACE
    ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS
    key = f"optillm:{ns}:cache:{entry_id}"

    data = {
        "prompt_text": prompt_text,
        "embedding": query_embedding.tolist(),
        "response_text": response_text,
        "model": model,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
    }

    try:
        client.setex(key, ttl, json.dumps(data))
        logger.info("Inserted entry into Redis cache | key=%s | ttl=%ds", key, ttl)
        return True
    except Exception as exc:
        logger.warning("Failed to insert into Redis cache (%s)", exc)
        return False


def clear_redis_cache(namespace: Optional[str] = None) -> int:
    """Clears all Redis cache keys under specified namespace."""
    client = get_redis_client()
    if not client:
        return 0

    ns = namespace or settings.CACHE_NAMESPACE
    pattern = f"optillm:{ns}:cache:*"
    try:
        keys = client.keys(pattern)
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as exc:
        logger.warning("Failed to clear Redis cache (%s)", exc)
        return 0
