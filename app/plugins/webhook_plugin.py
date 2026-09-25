"""
Webhook Plugin for OptiLLM Gateway
Dispatches request and response execution events asynchronously to configured HTTP endpoints.
Supports HMAC-SHA256 signature verification for downstream security.
"""

import hashlib
import hmac
import json
import logging
import threading
from typing import Any, Dict, Optional
import urllib.request

from app.plugins.base import BasePlugin

logger = logging.getLogger("optillm.plugins.webhook")


class WebhookPlugin(BasePlugin):
    """
    Enterprise Webhook Plugin:
    Emits asynchronous HTTP POST notifications on completion events without blocking client latency.
    """

    def __init__(
        self,
        name: str = "webhook",
        enabled: bool = True,
        webhook_url: Optional[str] = None,
        secret_token: Optional[str] = None,
    ):
        super().__init__(name=name, enabled=enabled)
        self.webhook_url = webhook_url
        self.secret_token = secret_token

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self.webhook_url = config.get("webhook_url", self.webhook_url)
            self.secret_token = config.get("secret_token", self.secret_token)
            self.enabled = config.get("enabled", self.enabled)

    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through pre_process unmodified."""
        return request_payload

    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches event payload asynchronously on completion."""
        if not self.webhook_url:
            return response_payload

        event_data = {
            "event": "llm.completion",
            "id": response_payload.get("id"),
            "model_requested": response_payload.get("model_requested"),
            "model_used": response_payload.get("model"),
            "cache_hit": response_payload.get("cache_hit", False),
            "routed": response_payload.get("routed", False),
            "latency_ms": response_payload.get("latency_ms", 0),
            "cost_usd": response_payload.get("cost_usd", 0.0),
            "savings_usd": response_payload.get("savings_usd", 0.0),
            "tokens_input": response_payload.get("tokens_input", 0),
            "tokens_output": response_payload.get("tokens_output", 0),
        }

        # Dispatch in detached background thread to maintain zero client latency overhead
        thread = threading.Thread(
            target=self._send_webhook,
            args=(self.webhook_url, event_data, self.secret_token),
            daemon=True,
        )
        thread.start()

        return response_payload

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Dispatches error notification webhook."""
        if not self.webhook_url:
            return

        error_event = {
            "event": "llm.error",
            "error_type": error.__class__.__name__,
            "message": str(error),
            "context": context,
        }
        thread = threading.Thread(
            target=self._send_webhook,
            args=(self.webhook_url, error_event, self.secret_token),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _send_webhook(url: str, payload: Dict[str, Any], secret_token: Optional[str] = None):
        try:
            body = json.dumps(payload).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "OptiLLM-Webhook-Dispatcher/1.0",
            }
            if secret_token:
                signature = hmac.new(
                    secret_token.encode("utf-8"), body, hashlib.sha256
                ).hexdigest()
                headers["X-OptiLLM-Signature"] = f"sha256={signature}"

            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                logger.debug("Webhook delivered to %s (status=%s)", url, resp.status)
        except Exception as e:
            logger.warning("Failed to dispatch webhook to %s: %s", url, e)
