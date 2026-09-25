"""
Langfuse Observability Callback.
Sends request traces to Langfuse for LLM observability, cost tracking, and quality monitoring.

Configure via environment variables:
  LANGFUSE_PUBLIC_KEY  — Your Langfuse public key
  LANGFUSE_SECRET_KEY  — Your Langfuse secret key
  LANGFUSE_HOST        — Langfuse host (default: https://cloud.langfuse.com)
"""

import logging
from typing import Any, Dict

import httpx

from app.core.callback_manager import BaseCallback
from app.core.config import settings

logger = logging.getLogger("optillm.callbacks.langfuse")


class LangfuseCallback(BaseCallback):
    """
    Sends LLM traces to Langfuse using the Generations API.
    Each request creates one Langfuse Generation trace with:
      - prompt, completion, model, tokens, cost, latency
      - cache_hit, compressed, routed metadata
    """

    @property
    def name(self) -> str:
        return "langfuse"

    def _auth(self):
        return (settings.LANGFUSE_PUBLIC_KEY, settings.LANGFUSE_SECRET_KEY)

    def _host(self) -> str:
        return settings.LANGFUSE_HOST.rstrip("/")

    async def async_log_success(self, payload: Dict[str, Any]) -> None:
        """Posts a generation trace to Langfuse."""
        import time

        messages = payload.get("messages", [])
        input_content = messages[-1].get("content", "") if messages else ""

        body = {
            "name": "optillm-completion",
            "model": payload.get("model", "unknown"),
            "modelParameters": {
                "temperature": payload.get("temperature", 0.7),
                "maxTokens": payload.get("max_tokens"),
            },
            "input": input_content[:2000],  # truncate for Langfuse
            "output": payload.get("content", "")[:2000],
            "usage": {
                "input": payload.get("tokens_input", 0),
                "output": payload.get("tokens_output", 0),
                "total": payload.get("tokens_input", 0) + payload.get("tokens_output", 0),
            },
            "metadata": {
                "cache_hit": payload.get("cache_hit", False),
                "compressed": payload.get("compressed", False),
                "routed": payload.get("routed", False),
                "model_requested": payload.get("model_requested"),
                "routing_reason": payload.get("routing_reason"),
                "latency_ms": payload.get("latency_ms", 0),
                "cost_usd": payload.get("cost_usd", 0.0),
                "savings_usd": payload.get("savings_usd", 0.0),
            },
            "startTime": payload.get("start_time_iso", ""),
            "endTime": payload.get("end_time_iso", ""),
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._host()}/api/public/generations",
                    json=body,
                    auth=self._auth(),
                )
                response.raise_for_status()
                logger.debug("Langfuse trace logged: status=%d", response.status_code)
        except Exception as exc:
            logger.warning("Langfuse callback failed: %s", exc)

    async def async_log_failure(self, payload: Dict[str, Any]) -> None:
        """Posts a failed generation trace to Langfuse."""
        body = {
            "name": "optillm-completion",
            "model": payload.get("model", "unknown"),
            "input": str(payload.get("messages", []))[:500],
            "output": None,
            "level": "ERROR",
            "statusMessage": payload.get("error", "Unknown error"),
            "metadata": {"error": payload.get("error")},
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._host()}/api/public/generations",
                    json=body,
                    auth=self._auth(),
                )
                response.raise_for_status()
        except Exception as exc:
            logger.warning("Langfuse failure callback error: %s", exc)
