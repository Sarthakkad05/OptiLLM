"""
Ollama Local Provider Adapter.
Communicates with local Ollama chat API (http://localhost:11434/api/chat).
Implements the BaseProvider interface with zero API cost ($0.00).
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.providers.base import BaseProvider
from app.services.token_counter import count_tokens_in_string

# Cache Ollama reachability result to avoid checking on every request
_ollama_available_cache: Optional[bool] = None
_ollama_last_check: float = 0.0
_OLLAMA_CHECK_INTERVAL = 30.0  # re-check every 30 seconds

logger = logging.getLogger("optillm.provider.ollama")

_OLLAMA_MODELS = {
    "llama3",
    "llama3.1",
    "llama3.2",
    "mistral",
    "qwen",
    "phi3",
    "gemma",
    "codellama",
    "vicuna",
}


class OllamaProvider(BaseProvider):
    """Ollama Local Provider Adapter implementing BaseProvider contract."""

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Returns True only if Ollama server is reachable at configured URL."""
        global _ollama_available_cache, _ollama_last_check

        if not settings.OLLAMA_BASE_URL:
            return False

        now = time.time()
        if (
            _ollama_available_cache is not None
            and now - _ollama_last_check < _OLLAMA_CHECK_INTERVAL
        ):
            return _ollama_available_cache

        try:
            base_url = settings.OLLAMA_BASE_URL.rstrip("/")
            resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
            _ollama_available_cache = resp.status_code == 200
        except Exception:
            _ollama_available_cache = False

        _ollama_last_check = now
        if not _ollama_available_cache:
            logger.debug(
                "Ollama server not reachable at %s — marking unavailable.",
                settings.OLLAMA_BASE_URL,
            )
        return _ollama_available_cache

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        clean_name = model_lower.replace("ollama/", "")
        return clean_name in _OLLAMA_MODELS or model_lower.startswith("ollama/")

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        api_url = f"{base_url}/api/chat"
        clean_model = model.replace("ollama/", "")

        payload = {
            "model": clean_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        logger.info("Calling Ollama | model=%s | messages=%d", clean_model, len(messages))
        start = time.time()

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()

        elapsed_ms = int((time.time() - start) * 1000)
        data = response.json()

        content = data.get("message", {}).get("content", "")
        tokens_in = data.get("prompt_eval_count", 0)
        tokens_out = data.get("eval_count", 0)

        if tokens_in == 0:
            all_text = " ".join(m.get("content", "") for m in messages)
            tokens_in = count_tokens_in_string(all_text, model="gpt-4o")
        if tokens_out == 0:
            tokens_out = count_tokens_in_string(content, model="gpt-4o")

        logger.info(
            "Ollama response | latency=%dms | tokens_in=%d | tokens_out=%d",
            elapsed_ms,
            tokens_in,
            tokens_out,
        )

        return {
            "id": f"ollama-{uuid.uuid4().hex[:8]}",
            "content": content,
            "model": clean_model,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "finish_reason": "stop",
            "latency_ms": elapsed_ms,
            "provider": "ollama",
        }

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        api_url = f"{base_url}/api/chat"
        clean_model = model.replace("ollama/", "")

        payload = {
            "model": clean_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        logger.info("Streaming Ollama | model=%s | messages=%d", clean_model, len(messages))

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", api_url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            delta = chunk.get("message", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue


# Instantiated Ollama Provider singleton
ollama_provider = OllamaProvider()


async def call_ollama(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return await ollama_provider.call(messages, model, temperature, max_tokens)


async def stream_ollama(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    async for chunk in ollama_provider.stream(messages, model, temperature, max_tokens):
        yield chunk
