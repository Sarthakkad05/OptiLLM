"""
OpenAI Provider Adapter.
Sends requests to the OpenAI Chat Completions API using httpx.
Implements the BaseProvider interface.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.providers.base import BaseProvider

logger = logging.getLogger("optillm.provider.openai")

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
_PLACEHOLDER_KEYS = {"your_openai_api_key_here", "", None}


class OpenAIProvider(BaseProvider):
    """OpenAI Provider Adapter implementing BaseProvider contract."""

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return settings.OPENAI_API_KEY not in _PLACEHOLDER_KEYS

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        return (
            model_lower.startswith("gpt")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or "openai" in model_lower
        )

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
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

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
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

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available():
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

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
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
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except Exception:
                            continue

    async def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
    ) -> List[List[float]]:
        """Generate embeddings via OpenAI Embeddings API."""
        if not self.is_available():
            raise ValueError("OPENAI_API_KEY is not set in .env")

        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"input": texts, "model": model}

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings", json=payload, headers=headers
            )
            response.raise_for_status()

        data = response.json()
        return [item["embedding"] for item in data["data"]]


# Instantiated OpenAI Provider
openai_provider = OpenAIProvider()


# Backward-compatible function wrappers
async def call_openai(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return await openai_provider.call(messages, model, temperature, max_tokens)


async def stream_openai(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    async for chunk in openai_provider.stream(messages, model, temperature, max_tokens):
        yield chunk
