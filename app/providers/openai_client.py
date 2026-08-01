"""
OpenAI Provider Client
Sends requests to the OpenAI Chat Completions API using httpx.
Returns a normalised internal response dict.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger("optillm.provider.openai")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


async def call_openai(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calls OpenAI Chat Completions API.
    Returns a normalised dict with: content, tokens_input, tokens_output, model.
    Raises httpx.HTTPStatusError on provider error.
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info("Calling OpenAI | model=%s | messages=%d", model, len(messages))
    start = time.time()

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OPENAI_API_URL, json=payload, headers=headers)
        response.raise_for_status()

    elapsed_ms = int((time.time() - start) * 1000)
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    logger.info(
        "OpenAI response | latency=%dms | tokens_in=%d | tokens_out=%d",
        elapsed_ms,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )

    return {
        "id": data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
        "content": content,
        "model": data.get("model", model),
        "tokens_input": usage.get("prompt_tokens", 0),
        "tokens_output": usage.get("completion_tokens", 0),
        "finish_reason": data["choices"][0].get("finish_reason", "stop"),
        "latency_ms": elapsed_ms,
        "provider": "openai",
    }


async def stream_openai(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    """
    Streams OpenAI response chunks via SSE.
    Yields content delta text strings.
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    logger.info("Streaming OpenAI | model=%s | messages=%d", model, len(messages))

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", OPENAI_API_URL, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        import json

                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except Exception:
                        continue
