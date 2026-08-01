import pytest


@pytest.mark.asyncio
async def test_anthropic_chat_completions_mock_mode(async_client):
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello, Claude test!"}],
    }
    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "optillm_metadata" in data
    assert data["optillm_metadata"]["model_requested"] == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_anthropic_chat_completions_streaming(async_client):
    payload = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hello, stream Claude!"}],
        "stream": True,
    }
    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    content = response.text
    assert "data: " in content
    assert "[DONE]" in content
