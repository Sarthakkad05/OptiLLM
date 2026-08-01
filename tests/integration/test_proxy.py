import pytest


@pytest.mark.asyncio
async def test_chat_completions_mock_mode(async_client):
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello,OptiLLM!"}],
    }
    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "optillm_metadata" in data
    assert data["optillm_metadata"]["model_requested"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_chat_completions_streaming(async_client):
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello, stream test!"}],
        "stream": True,
    }
    response = await async_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    content = response.text
    assert "data: " in content
    assert "[DONE]" in content
