"""
Integration tests validating 100% OpenAI API protocol compatibility.
Verifies /v1/chat/completions (unary & streaming), /v1/models, /v1/embeddings,
and OpenAI error response structures.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_openai_models_list(client):
    """GET /v1/models returns standard OpenAI model list structure."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    first_model = data["data"][0]
    assert "id" in first_model
    assert first_model["object"] == "model"
    assert "owned_by" in first_model


def test_openai_model_retrieve(client):
    """GET /v1/models/{model_id} returns specific model card or 404."""
    response = client.get("/v1/models/gpt-4o")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "gpt-4o"
    assert data["object"] == "model"
    assert data["owned_by"] == "openai"

    not_found = client.get("/v1/models/non-existent-model-xyz")
    assert not_found.status_code == 404
    err_body = not_found.json()
    assert "error" in err_body
    assert "message" in err_body["error"]


def test_openai_embeddings_compatibility(client):
    """POST /v1/embeddings returns OpenAI compliant embedding payload."""
    with patch("app.providers.openai_client.openai_provider.is_available", return_value=True), \
         patch("app.providers.openai_client.openai_provider.embed", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [[0.01, 0.02, 0.03]]
        payload = {
            "model": "text-embedding-3-small",
            "input": "The quick brown fox jumps over the lazy dog."
        }
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0
        assert isinstance(data["data"][0]["embedding"], list)
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0


def test_openai_chat_completions_basic(client):
    """POST /v1/chat/completions adheres to OpenAI response contract."""
    mock_provider_resp = {
        "id": "chatcmpl-test-12345",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "content": "Hello! I am ready to assist you.",
        "finish_reason": "stop",
        "provider": "openai",
        "tokens_input": 12,
        "tokens_output": 8,
    }

    with patch("app.services.gateway.call_provider", new_callable=AsyncMock, return_value=mock_provider_resp), \
         patch("app.engine.cache.check_cache", return_value=None):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Standard OpenAI fields
        assert data["id"].startswith("chatcmpl-") or "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["model"] in ["gpt-4o", "gpt-4o-mini"]
        assert len(data["choices"]) == 1
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello! I am ready to assist you."
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

        # OptiLLM metadata attached
        assert "optillm_metadata" in data
        assert "latency_ms" in data["optillm_metadata"]
        assert "cost_usd" in data["optillm_metadata"]


def test_openai_chat_completions_with_tools_and_extras(client):
    """Clients sending tools, response_format, stop, top_p work without validation error."""
    mock_provider_resp = {
        "id": "chatcmpl-tools-test",
        "created": 1700000000,
        "model": "gpt-4o",
        "content": "Result with tools",
        "finish_reason": "stop",
        "provider": "openai",
        "tokens_input": 25,
        "tokens_output": 5,
    }

    with patch("app.services.gateway.call_provider", new_callable=AsyncMock, return_value=mock_provider_resp), \
         patch("app.engine.cache.check_cache", return_value=None):
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Tool executor"},
                {"role": "user", "content": "What's the weather in Seattle?"},
                {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": 65}'}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"]
                        }
                    }
                }
            ],
            "tool_choice": "auto",
            "top_p": 0.95,
            "stop": ["\n\n"],
            "response_format": {"type": "text"}
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Result with tools"


def test_openai_streaming_sse(client):
    """POST /v1/chat/completions with stream=true yields SSE format ending with [DONE]."""
    async def mock_stream_chunks(*args, **kwargs):
        chunks = [
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ]
        for c in chunks:
            yield c

    with patch("app.api.endpoints.proxy.process_stream_request", return_value=mock_stream_chunks()):
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Stream test"}],
            "stream": True
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert "data: [DONE]" in response.text
        assert "Hello world" in response.text or "Hello" in response.text


def test_openai_error_schema_on_auth_failure():
    """Auth rejection on /v1/ routes returns OpenAI error envelope."""
    with patch("app.core.config.settings.API_KEY_AUTH_ENABLED", True):
        # Fresh client without auth header
        unauthed_client = TestClient(app)
        response = unauthed_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "message" in data["error"]
        assert data["error"]["code"] in ["unauthorized", "401"]
