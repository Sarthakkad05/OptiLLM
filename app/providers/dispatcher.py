"""
Provider Dispatcher.
Routes completion requests dynamically to registered BaseProvider adapters using ProviderRegistry.

Features:
  - Dynamic Provider Lookup via ProviderRegistry
  - Automatic Retry with exponential backoff on transient errors
  - Provider Fallback to available alternative providers if primary fails
  - Mock Mode when no provider API keys are configured
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.providers.registry import provider_registry
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.provider.dispatcher")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — doubled on each attempt


def _is_mock_mode() -> bool:
    """Return True if no registered provider has valid credentials."""
    return len(provider_registry.list_available_providers()) == 0


async def _mock_response(messages: List[Dict], model: str) -> Dict[str, Any]:
    """
    Returns a realistic mock LLM response for local testing when no API keys are set.
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    last_user_msg = (
        user_messages[-1].get("content", "Hello") if user_messages else "Hello"
    )

    mock_answer = (
        f'[MOCK] Simulated answer to: "{last_user_msg[:80]}..."\n\n'
        f"In production with valid API keys, this would be a real response from {model}. "
        f"The OptiLLM optimization pipeline ran successfully."
    )

    tokens_in = count_tokens_in_messages(messages, model)
    tokens_out = len(mock_answer.split()) + 10

    await asyncio.sleep(0.05)

    logger.info(
        "MOCK RESPONSE | model=%s | tokens_in=%d | tokens_out=%d",
        model,
        tokens_in,
        tokens_out,
    )

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
    provider_adapter,
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
) -> Dict[str, Any]:
    """Call provider adapter with exponential backoff retry."""
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return await provider_adapter.call(messages, model, temperature, max_tokens)
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Provider %s attempt %d/%d failed: %s — retrying in %.1fs",
                    provider_adapter.name,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Provider %s failed after %d attempts: %s",
                    provider_adapter.name,
                    _MAX_RETRIES,
                    exc,
                )
    raise last_exc


async def call_provider(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dispatches request to appropriate registered provider with retry and fallback.
    """
    if _is_mock_mode():
        logger.warning("No API keys configured — MOCK MODE enabled.")
        return await _mock_response(messages, model)

    # Resolve target provider using registry
    provider = provider_registry.get_for_model(model)
    if not provider:
        # Fallback default provider
        provider = provider_registry.get("openai") or provider_registry.get("gemini")

    if provider and provider.is_available():
        try:
            return await _call_with_retry(
                provider, messages, model, temperature, max_tokens
            )
        except Exception as primary_exc:
            logger.warning(
                "Primary provider '%s' failed: %s — trying fallback provider.",
                provider.name,
                primary_exc,
            )

    # Fallback to any other available provider
    for alt_name in provider_registry.list_available_providers():
        alt_provider = provider_registry.get(alt_name)
        if alt_provider and alt_provider != provider:
            fallback_model = (
                "gpt-4o-mini" if alt_name == "openai" else "gemini-2.0-flash"
            )
            logger.info(
                "Falling back to provider '%s' with model '%s'",
                alt_name,
                fallback_model,
            )
            res = await _call_with_retry(
                alt_provider, messages, fallback_model, temperature, max_tokens
            )
            res["provider"] = f"{alt_name}-fallback"
            return res

    raise ValueError(
        f"No available provider for model '{model}'. "
        "Check OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY in .env"
    )


async def stream_provider(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    """
    Streams completion chunks from the target provider adapter via registry lookup.
    """
    if _is_mock_mode():
        user_messages = [m for m in messages if m.get("role") == "user"]
        last_msg = (
            user_messages[-1].get("content", "Hello") if user_messages else "Hello"
        )
        mock_tokens = [
            "[MOCK] ",
            "Simulated ",
            "streamed ",
            "response ",
            "to: ",
            f'"{last_msg[:40]}..."\n',
            "OptiLLM ",
            "stream ",
            "pipeline ",
            "working ",
            "cleanly!",
        ]
        for token in mock_tokens:
            await asyncio.sleep(0.02)
            yield token
        return

    provider = provider_registry.get_for_model(model)
    if provider and provider.is_available():
        async for chunk in provider.stream(messages, model, temperature, max_tokens):
            yield chunk
        return

    # Fallback to any other available provider with a compatible model
    for alt_name in provider_registry.list_available_providers():
        alt_provider = provider_registry.get(alt_name)
        if alt_provider:
            fallback_model = (
                "gpt-4o-mini"
                if alt_name == "openai"
                else (
                    "gemini-2.0-flash"
                    if alt_name == "gemini"
                    else "claude-3-5-haiku-20241022"
                )
            )
            logger.info(
                "Streaming fallback to provider '%s' with model '%s'",
                alt_name,
                fallback_model,
            )
            async for chunk in alt_provider.stream(
                messages, fallback_model, temperature, max_tokens
            ):
                yield chunk
            return

    raise ValueError("No available provider to stream.")
