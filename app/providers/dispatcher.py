"""
Provider Dispatcher.
Routes completion requests dynamically using ProviderRegistry, CircuitBreaker, and LoadBalancer.

Features:
  - Load Balancer Strategy Selection (cost_optimized, round_robin, least_latency)
  - Circuit Breaker Fault Isolation (CLOSED, OPEN, HALF_OPEN)
  - Automatic Retry with exponential backoff on transient errors
  - Automatic Provider Fallback if primary fails or circuit is OPEN
  - Mock Mode when no provider API keys are configured
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.core.circuit_breaker import circuit_breaker
from app.providers.load_balancer import load_balancer
from app.providers.registry import provider_registry
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.provider.dispatcher")

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds — doubled on each attempt

# Per-provider timeout overrides (seconds).
# Falls back to settings.REQUEST_TIMEOUT_SECONDS if provider not listed.
_PROVIDER_TIMEOUTS: Dict[str, float] = {
    "openai": 30.0,
    "gemini": 30.0,
    "anthropic": 90.0,   # Claude models are slower by nature
    "groq": 20.0,        # Groq is very fast; short timeout is safe
    "mistral": 45.0,
    "azure": 30.0,
    "bedrock": 60.0,
    "ollama": 300.0,     # Local models have no SLA; allow generous time
}


def _get_provider_timeout(provider_name: str) -> float:
    """Return the configured timeout for a provider, falling back to global setting."""
    from app.core.config import settings
    return _PROVIDER_TIMEOUTS.get(
        (provider_name or "").lower(), settings.REQUEST_TIMEOUT_SECONDS
    )


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
    """Call provider adapter with exponential backoff retry, circuit breaker, and per-provider timeout."""
    timeout = _get_provider_timeout(getattr(provider_adapter, "name", ""))
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            res = await asyncio.wait_for(
                provider_adapter.call(messages, model, temperature, max_tokens),
                timeout=timeout,
            )
            circuit_breaker.record_success(
                provider_adapter.name, res.get("latency_ms", 0.0)
            )
            return res
        except asyncio.TimeoutError:
            last_exc = TimeoutError(
                f"Provider '{provider_adapter.name}' timed out after {timeout}s "
                f"(attempt {attempt}/{_MAX_RETRIES})"
            )
            circuit_breaker.record_failure(provider_adapter.name)
            logger.warning(str(last_exc))
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        except Exception as exc:
            last_exc = exc
            circuit_breaker.record_failure(provider_adapter.name)
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
    Dispatches request using LoadBalancer & CircuitBreaker.
    """
    if _is_mock_mode():
        logger.warning("No API keys configured — MOCK MODE enabled.")
        return await _mock_response(messages, model)

    # Select provider via LoadBalancer (respects circuit breaker state)
    provider = load_balancer.select_provider(model=model)

    if (
        provider
        and provider.is_available()
        and circuit_breaker.can_execute(provider.name)
    ):
        try:
            return await _call_with_retry(
                provider, messages, model, temperature, max_tokens
            )
        except Exception as primary_exc:
            logger.warning(
                "Primary provider '%s' failed: %s — attempting fallback provider.",
                provider.name,
                primary_exc,
            )

    # Fallback to any other healthy available provider
    for alt_name in provider_registry.list_available_providers():
        if circuit_breaker.can_execute(alt_name):
            alt_provider = provider_registry.get(alt_name)
            if alt_provider and alt_provider != provider:
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
                    "Falling back to healthy provider '%s' with model '%s'",
                    alt_name,
                    fallback_model,
                )
                try:
                    res = await _call_with_retry(
                        alt_provider, messages, fallback_model, temperature, max_tokens
                    )
                    res["provider"] = f"{alt_name}-fallback"
                    return res
                except Exception as fallback_exc:
                    logger.warning(
                        "Fallback provider '%s' failed: %s", alt_name, fallback_exc
                    )

    from app.core.config import settings
    if settings.APP_ENV == "production":
        logger.error("All providers failed or circuit-broken in production — returning 503.")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="All AI providers are currently unavailable or circuit-broken. Please retry later.",
            headers={"Retry-After": "30"},
        )

    logger.warning("All primary/fallback providers failed — falling back to mock mode.")
    return await _mock_response(messages, model)


async def stream_provider(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
):
    """
    Streams completion chunks from healthy provider using LoadBalancer & CircuitBreaker.
    """

    async def _yield_mock_stream():
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

    if _is_mock_mode():
        async for token in _yield_mock_stream():
            yield token
        return

    provider = load_balancer.select_provider(model=model)
    if (
        provider
        and provider.is_available()
        and circuit_breaker.can_execute(provider.name)
    ):
        try:
            async for chunk in provider.stream(
                messages, model, temperature, max_tokens
            ):
                yield chunk
            circuit_breaker.record_success(provider.name)
            return
        except Exception as exc:
            circuit_breaker.record_failure(provider.name)
            logger.warning(
                "Streaming provider '%s' failed: %s — trying fallback.",
                provider.name,
                exc,
            )

    # Fallback streaming to any other healthy available provider
    for alt_name in provider_registry.list_available_providers():
        if circuit_breaker.can_execute(alt_name):
            alt_provider = provider_registry.get(alt_name)
            if alt_provider and alt_provider != provider:
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
                    "Streaming fallback to healthy provider '%s' with model '%s'",
                    alt_name,
                    fallback_model,
                )
                try:
                    async for chunk in alt_provider.stream(
                        messages, fallback_model, temperature, max_tokens
                    ):
                        yield chunk
                    circuit_breaker.record_success(alt_name)
                    return
                except Exception as fb_exc:
                    circuit_breaker.record_failure(alt_name)
                    logger.warning(
                        "Streaming fallback '%s' failed: %s", alt_name, fb_exc
                    )

    from app.core.config import settings
    if settings.APP_ENV == "production":
        logger.error("All streaming providers failed or circuit-broken in production.")
        import json
        yield f"data: {json.dumps({'error': {'message': 'All AI providers are currently unavailable or circuit-broken.', 'type': 'service_unavailable', 'code': 503}})}\n\n"
        return

    logger.warning("All streaming providers failed — falling back to mock stream.")
    async for token in _yield_mock_stream():
        yield token
