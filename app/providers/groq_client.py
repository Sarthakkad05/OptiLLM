"""
Groq Provider Adapter.
Groq uses an OpenAI-compatible API surface — just a different base URL and API key.
Supports Llama 3, Mixtral, and Gemma models with ultra-low latency inference.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.providers.base import BaseProvider

logger = logging.getLogger("optillm.provider.groq")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_PLACEHOLDER_KEYS = {"", None}

_GROQ_MODELS = {
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
}


class GroqProvider(BaseProvider):
    """Groq Provider Adapter — ultra-fast inference via OpenAI-compatible API."""

    @property
    def name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        return settings.GROQ_API_KEY not in _PLACEHOLDER_KEYS

    def supports_model(self, model: str) -> bool:
        return model.lower() in _GROQ_MODELS or "llama" in model.lower() or "mixtral" in model.lower()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("GROQ_API_KEY is not set in .env")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info("Calling Groq | model=%s | messages=%d", model, len(messages))
        start = time.time()

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()

        elapsed_ms = int((time.time() - start) * 1000)
        data = response.json()
        usage = data.get("usage", {})

        return {
            "id": data.get("id", f"groq-{uuid.uuid4().hex[:8]}"),
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "tokens_input": usage.get("prompt_tokens", 0),
            "tokens_output": usage.get("completion_tokens", 0),
            "finish_reason": data["choices"][0].get("finish_reason", "stop"),
            "latency_ms": elapsed_ms,
            "provider": "groq",
        }

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available():
            raise ValueError("GROQ_API_KEY is not set in .env")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST", f"{GROQ_BASE_URL}/chat/completions",
                json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            if content:
                                yield content
                        except Exception:
                            continue


groq_provider = GroqProvider()
