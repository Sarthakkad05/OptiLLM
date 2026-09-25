"""
Slack Webhook Callback.
Sends alerts to a Slack channel for budget overruns and provider failures.

Configure via: SLACK_WEBHOOK_URL env var
"""

import logging
from typing import Any, Dict

import httpx

from app.core.callback_manager import BaseCallback
from app.core.config import settings

logger = logging.getLogger("optillm.callbacks.slack")


class SlackCallback(BaseCallback):
    """Sends budget and failure alerts to a Slack webhook."""

    @property
    def name(self) -> str:
        return "slack"

    async def async_log_success(self, payload: Dict[str, Any]) -> None:
        """Only fires for budget alerts (high cost requests)."""
        cost = payload.get("cost_usd", 0.0)
        # Only alert on unusually expensive single requests (>$0.10)
        if cost > 0.10:
            await self._send_message(
                f":moneybag: *High-cost request detected*\n"
                f"• Model: `{payload.get('model', 'unknown')}`\n"
                f"• Cost: `${cost:.4f}`\n"
                f"• Tokens: `{payload.get('tokens_input', 0)}` in / `{payload.get('tokens_output', 0)}` out\n"
                f"• Latency: `{payload.get('latency_ms', 0)}ms`"
            )

    async def async_log_failure(self, payload: Dict[str, Any]) -> None:
        """Fires on every provider failure."""
        await self._send_message(
            f":x: *OptiLLM Provider Failure*\n"
            f"• Model: `{payload.get('model', 'unknown')}`\n"
            f"• Error: `{str(payload.get('error', 'Unknown'))[:200]}`"
        )

    async def _send_message(self, text: str) -> None:
        if not settings.SLACK_WEBHOOK_URL:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    settings.SLACK_WEBHOOK_URL,
                    json={"text": text},
                )
                response.raise_for_status()
                logger.debug("Slack alert sent.")
        except Exception as exc:
            logger.warning("Slack callback failed: %s", exc)
