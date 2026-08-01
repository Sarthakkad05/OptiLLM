"""
Anthropic Provider Adapter.
Communicates with Anthropic Messages API (Claude 3 family).
Implements the BaseProvider interface.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.providers.base import BaseProvider
from app.services.token_counter import count_tokens_in_string

logger = logging.getLogger("optillm.provider.anthropic")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_PLACEHOLDER_KEYS = {"your_anthropic_api_key_here", "", None}

_ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "claude-3-opus",
}


def _openai_messages_to_anthropic(
    messages: List[Dict],
) -> Tuple[Optional[str], List[Dict]]:
    """
    Extract system prompt and convert messages to Anthropic's format.
    Anthropic requires system prompt as a top-level parameter.
    """
    system_prompt = None
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            if system_prompt:
                system_prompt += f"\n\n{content}"
            else:
                system_prompt = content
            continue

        anthropic_role = "user" if role == "user" else "assistant"
        anthropic_messages.append({"role": anthropic_role, "content": content})

    return system_prompt, anthropic_messages


class AnthropicProvider(BaseProvider):
    """Anthropic Provider Adapter implementing BaseProvider contract."""

    @property
    def name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        return settings.ANTHROPIC_API_KEY not in _PLACEHOLDER_KEYS

    def supports_model(self, model: str) -> bool:
        model_lower = model.lower()
        return model_lower in _ANTHROPIC_MODELS or model_lower.startswith("claude")

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")

        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": settings.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        logger.info("Calling Anthropic | model=%s | messages=%d", model, len(messages))
        start = time.time()

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                ANTHROPIC_API_URL, json=payload, headers=headers
            )
            response.raise_for_status()

        elapsed_ms = int((time.time() - start) * 1000)
        data = response.json()

        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        if tokens_in == 0:
            all_text = " ".join(m.get("content", "") for m in messages)
            tokens_in = count_tokens_in_string(all_text, model="gpt-4o")
        if tokens_out == 0:
            tokens_out = count_tokens_in_string(content, model="gpt-4o")

        logger.info(
            "Anthropic response | latency=%dms | tokens_in=%d | tokens_out=%d",
            elapsed_ms,
            tokens_in,
            tokens_out,
        )

        return {
            "id": data.get("id", f"msg_{uuid.uuid4().hex[:8]}"),
            "content": content,
            "model": data.get("model", model),
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "finish_reason": data.get("stop_reason", "end_turn"),
            "latency_ms": elapsed_ms,
            "provider": "anthropic",
        }

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available():
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")

        system_prompt, anthropic_messages = _openai_messages_to_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": settings.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        logger.info(
            "Streaming Anthropic | model=%s | messages=%d", model, len(messages)
        )

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", ANTHROPIC_API_URL, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("type") == "content_block_delta":
                                delta_text = chunk.get("delta", {}).get("text")
                                if delta_text:
                                    yield delta_text
                        except Exception:
                            continue


# Instantiated Anthropic Provider
anthropic_provider = AnthropicProvider()


# Function wrappers for provider calls
async def call_anthropic(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return await anthropic_provider.call(messages, model, temperature, max_tokens)


async def stream_anthropic(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    async for chunk in anthropic_provider.stream(
        messages, model, temperature, max_tokens
    ):
        yield chunk
