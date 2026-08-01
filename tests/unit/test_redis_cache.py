import numpy as np

from app.engine.redis_cache import (
    clear_redis_cache,
    insert_redis_cache,
    is_redis_available,
    lookup_redis_cache,
)


def test_redis_available_returns_bool():
    res = is_redis_available()
    assert isinstance(res, bool)


def test_redis_cache_fallback_when_offline():
    # Vector embedding (384d)
    vector = np.random.rand(384).astype(np.float32)

    # Lookup should return None gracefully without raising exception when Redis is not connected
    match = lookup_redis_cache(vector, namespace="test_ns")
    assert match is None or isinstance(match, dict)


def test_redis_cache_insert_and_clear():
    vector = np.random.rand(384).astype(np.float32)
    ok = insert_redis_cache(
        entry_id="test_1",
        prompt_text="What is OptiLLM?",
        query_embedding=vector,
        response_text="OptiLLM is an AI Gateway.",
        model="gpt-4o",
        tokens_input=10,
        tokens_output=15,
        namespace="test_ns",
        ttl_seconds=60,
    )
    assert isinstance(ok, bool)

    cleared = clear_redis_cache(namespace="test_ns")
    assert isinstance(cleared, int)
