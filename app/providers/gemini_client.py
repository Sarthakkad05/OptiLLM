"""
Google Gemini Provider Adapter.
Uses Google's REST API (generateContent) and normalises the response
into the standard BaseProvider dict structure.
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

logger = logging.getLogger("optillm.provider.gemini")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_PLACEHOLDER_KEYS = {"your_gemini_api_key_here", "", None}


def _openai_messages_to_gemini(messages: List[Dict]) -> Dict:
    """
    Convert OpenAI message format to Gemini's contents format.
    System messages are prepended as the first user turn.
    """
    contents = []
    system_text = None

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_text = content
            continue

        gemini_role = "user" if role == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    # Prepend system message to first user turn
    if system_text and contents:
        first_text = contents[0]["parts"][0]["text"]
        contents[0]["parts"][0]["text"] = f"{system_text}\n\n{first_text}"

    return contents


class GeminiProvider(BaseProvider):
    """Gemini Provider Adapter implementing BaseProvider contract."""

    @property
    def name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return settings.GEMINI_API_KEY not in _PLACEHOLDER_KEYS

    def supports_model(self, model: str) -> bool:
        return model.lower().startswith("gemini")

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("GEMINI_API_KEY is not set in .env")

        contents = _openai_messages_to_gemini(messages)

        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        url = f"{GEMINI_API_BASE}/{model}:generateContent?key={settings.GEMINI_API_KEY}"

        logger.info("Calling Gemini | model=%s | turns=%d", model, len(contents))
        start = time.time()

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

        elapsed_ms = int((time.time() - start) * 1000)
        data = response.json()

        # Extract content
        candidate = data["candidates"][0]
        content = candidate["content"]["parts"][0]["text"]

        # Gemini returns usageMetadata
        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0)
        tokens_out = usage.get("candidatesTokenCount", 0)

        # Fallback: estimate tokens if not provided
        if tokens_in == 0:
            all_text = " ".join(m.get("content", "") for m in messages)
            tokens_in = count_tokens_in_string(all_text, model="gpt-4o")
        if tokens_out == 0:
            tokens_out = count_tokens_in_string(content, model="gpt-4o")

        logger.info(
            "Gemini response | latency=%dms | tokens_in=%d | tokens_out=%d",
            elapsed_ms,
            tokens_in,
            tokens_out,
        )

        return {
            "id": f"gemini-{uuid.uuid4().hex[:8]}",
            "content": content,
            "model": model,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "finish_reason": candidate.get("finishReason", "STOP").lower(),
            "latency_ms": elapsed_ms,
            "provider": "gemini",
        }

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available():
            raise ValueError("GEMINI_API_KEY is not set in .env")

        contents = _openai_messages_to_gemini(messages)
        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens

        payload = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        url = f"{GEMINI_API_BASE}/{model}:streamGenerateContent?alt=sse&key={settings.GEMINI_API_KEY}"
        logger.info("Streaming Gemini | model=%s | turns=%d", model, len(contents))

        timeout = settings.REQUEST_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        try:
                            chunk = json.loads(data_str)
                            candidates = chunk.get("candidates", [])
                            if candidates:
                                parts = (
                                    candidates[0].get("content", {}).get("parts", [])
                                )
                                if parts and "text" in parts[0]:
                                    yield parts[0]["text"]
                        except Exception:
                            continue


# Instantiated Gemini Provider
gemini_provider = GeminiProvider()


# Backward-compatible function wrappers
async def call_gemini(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    return await gemini_provider.call(messages, model, temperature, max_tokens)


async def stream_gemini(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    async for chunk in gemini_provider.stream(messages, model, temperature, max_tokens):
        yield chunk
