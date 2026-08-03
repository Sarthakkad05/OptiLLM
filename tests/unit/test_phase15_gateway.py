"""
Unit & Integration Tests for Phase 15 — HTTP Gateway Optimization Suite
Tests per-request optillm configuration, namespaced semantic cache,
compression modes, and compression preview visualizer endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.engine.cache import check_cache, insert_cache
from app.engine.compressor import compress
from app.main import app
from app.schemas.chat import ChatCompletionRequest, OptiLLMConfig

client = TestClient(app)


def test_optillm_config_schema_per_request_fields():
    """Verify OptiLLMConfig schema accepts per-request parameters."""
    cfg = OptiLLMConfig(
        bypass_cache=False,
        cache_threshold=0.92,
        cache_namespace="test_namespace",
        ttl_seconds=3600,
        compression_mode="smart",
    )
    assert cfg.cache_threshold == 0.92
    assert cfg.cache_namespace == "test_namespace"
    assert cfg.ttl_seconds == 3600
    assert cfg.compression_mode == "smart"


def test_compressor_modes():
    """Test smart, aggressive, and minimal compression modes."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Hello world\n\n\nHello world\n"},
    ]

    # Minimal mode
    cleaned_minimal, stats_minimal = compress(messages, mode="minimal")
    assert "was_compressed" in stats_minimal
    assert len(cleaned_minimal) == 2

    # Smart mode
    cleaned_smart, stats_smart = compress(messages, mode="smart")
    assert len(cleaned_smart) == 2

    # Aggressive mode
    cleaned_aggr, stats_aggr = compress(messages, mode="aggressive")
    assert len(cleaned_aggr) == 2


def test_compression_preview_endpoint():
    """Test POST /api/v1/compressor/preview endpoint."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "Write a python function to add two numbers."},
        ],
        "model": "gpt-4o",
        "mode": "smart",
        "max_tokens": 2000,
    }
    response = client.post("/api/v1/compressor/preview", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "original_messages" in data
    assert "compressed_messages" in data
    assert "stats" in data
    assert data["stats"]["original_tokens"] > 0
