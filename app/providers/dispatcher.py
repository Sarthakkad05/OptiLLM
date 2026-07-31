"""
Provider Dispatcher
Determines which provider client to call based on the model name.
All callers use call_provider() — never call openai/gemini clients directly.

Reliability:
  - Automatic retry with exponential backoff (3 attempts) on transient errors
  - Provider fallback: if the primary provider fails, fall back to the other
  - Mock mode: if no API keys are set, returns a realistic simulated response
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional

from app.providers.openai_client import call_openai
from app.providers.gemini_client import call_gemini
from app.core.config import settings
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.provider.dispatcher")

# Models that route to Gemini
_GEMINI_MODELS = {
    "gemini-1.5-pro", "gemini-1.5-flash",
    "gemini-2.0-flash", "gemini-1.0-pro", "gemini-pro",
}

_PLACEHOLDER_KEYS = {"your_openai_api_key_here", "your_gemini_api_key_here", "", None}

# Retry config
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0   # seconds — doubled on each attempt


def _detect_provider(model: str) -> str:
    """Returns 'gemini' or 'openai' based on the model name."""
    if model.lower() in _GEMINI_MODELS or model.lower().startswith("gemini"):
        return "gemini"
    return "openai"


def _is_mock_mode() -> bool:
    """Return True if no valid API keys are configured."""
    openai_missing = settings.OPENAI_API_KEY in _PLACEHOLDER_KEYS
    gemini_missing = settings.GEMINI_API_KEY in _PLACEHOLDER_KEYS
    return openai_missing and gemini_missing


def _openai_available() -> bool:
    return settings.OPENAI_API_KEY not in _PLACEHOLDER_KEYS


def _gemini_available() -> bool:
    return settings.GEMINI_API_KEY not in _PLACEHOLDER_KEYS


async def _mock_response(messages: List[Dict], model: str) -> Dict[str, Any]:
    """
    Returns a realistic mock LLM response for local testing.
    Lets the full pipeline (routing, compression, caching, cost logging) run
    without real API keys.
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    last_user_msg = user_messages[-1].get("content", "Hello") if user_messages else "Hello"

    mock_answer = (
        f"[MOCK] Simulated answer to: \"{last_user_msg[:80]}...\"\n\n"
        f"In production with valid API keys, this would be a real response from {model}. "
        f"The OptiLLM optimization pipeline (cache, compression, routing) ran successfully."
    )

    tokens_in = count_tokens_in_messages(messages, model)
    tokens_out = len(mock_answer.split()) + 10

    await asyncio.sleep(0.05)   # Simulate minimal latency

    logger.info("MOCK RESPONSE | model=%s | tokens_in=%d | tokens_out=%d", model, tokens_in, tokens_out)

    return {
        "id": f"mock-{uuid.uuid4().hex[:8]}",
        "content": mock_answer,
        "model": model,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "finish_reason": "stop",
        "latency_ms": 50,
        "provider": "mock",
    }


async def _call_with_retry(
    fn,
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    provider_label: str,
) -> Dict[str, Any]:
    """
    Call a provider function with exponential backoff retry.
    Raises the last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await fn(messages, model, temperature, max_tokens)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Provider %s attempt %d/%d failed: %s — retrying in %.1fs",
                    provider_label, attempt, _MAX_RETRIES, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Provider %s failed after %d attempts: %s",
                    provider_label, _MAX_RETRIES, exc,
                )
    raise last_exc


async def call_provider(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dispatches the request to the correct LLM provider with retry + fallback.

    Priority:
      1. Mock mode if no keys configured.
      2. Primary provider (based on model name) with up to 3 retries.
      3. Fallback to the other provider if primary exhausts retries.
    """
    if _is_mock_mode():
        logger.warning(
            "No API keys configured — MOCK MODE. "
            "Set OPENAI_API_KEY or GEMINI_API_KEY in .env for real responses."
        )
        return await _mock_response(messages, model)

    primary_provider = _detect_provider(model)
    logger.info("Dispatching to provider=%s model=%s", primary_provider, model)

    # Primary attempt with retry
    try:
        if primary_provider == "gemini" and _gemini_available():
            return await _call_with_retry(call_gemini, messages, model, temperature, max_tokens, "gemini")
        elif primary_provider == "openai" and _openai_available():
            return await _call_with_retry(call_openai, messages, model, temperature, max_tokens, "openai")
    except Exception as primary_exc:
        logger.warning("Primary provider (%s) failed: %s — attempting fallback.", primary_provider, primary_exc)

        # Fallback to the other provider
        if primary_provider == "gemini" and _openai_available():
            fallback_model = "gpt-4o-mini"   # Use a capable but cheap fallback model
            logger.info("Falling back to OpenAI | model=%s", fallback_model)
            result = await _call_with_retry(call_openai, messages, fallback_model, temperature, max_tokens, "openai-fallback")
            result["provider"] = "openai-fallback"
            return result
        elif primary_provider == "openai" and _gemini_available():
            fallback_model = "gemini-2.0-flash"
            logger.info("Falling back to Gemini | model=%s", fallback_model)
            result = await _call_with_retry(call_gemini, messages, fallback_model, temperature, max_tokens, "gemini-fallback")
            result["provider"] = "gemini-fallback"
            return result
        else:
            raise primary_exc  # No fallback available

    # Shouldn't reach here, but handle edge case where no provider is available
    raise ValueError(
        f"No available provider for model '{model}'. "
        "Check OPENAI_API_KEY and GEMINI_API_KEY in .env"
    )
