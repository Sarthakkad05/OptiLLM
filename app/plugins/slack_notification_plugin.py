"""
Slack Notification Plugin for OptiLLM Gateway
Posts formatted alert notifications to Slack incoming webhooks for high-latency calls,
guardrail violations, and provider errors.
"""

import json
import logging
import threading
from typing import Any, Dict, Optional
import urllib.request

from app.plugins.base import BasePlugin

logger = logging.getLogger("optillm.plugins.slack")


class SlackNotificationPlugin(BasePlugin):
    """
    Alerting plugin that formats and dispatches Slack notifications
    when operational thresholds are breached.
    """

    def __init__(
        self,
        name: str = "slack_notifier",
        enabled: bool = True,
        webhook_url: Optional[str] = None,
        latency_threshold_ms: int = 5000,
        notify_on_cache_miss_only: bool = False,
    ):
        super().__init__(name=name, enabled=enabled)
        self.webhook_url = webhook_url
        self.latency_threshold_ms = latency_threshold_ms
        self.notify_on_cache_miss_only = notify_on_cache_miss_only

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self.webhook_url = config.get("webhook_url", self.webhook_url)
            self.latency_threshold_ms = config.get(
                "latency_threshold_ms", self.latency_threshold_ms
            )
            self.notify_on_cache_miss_only = config.get(
                "notify_on_cache_miss_only", self.notify_on_cache_miss_only
            )
            self.enabled = config.get("enabled", self.enabled)

    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return request_payload

    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.webhook_url:
            return response_payload

        latency_ms = response_payload.get("latency_ms", 0)
        is_slow = latency_ms >= self.latency_threshold_ms
        has_guardrails = bool(response_payload.get("guardrail_warnings"))

        if is_slow or has_guardrails:
            text = (
                f":warning: *OptiLLM Alert*\n"
                f"• *Model:* `{response_payload.get('model')}`\n"
                f"• *Latency:* `{latency_ms}ms`" + (" *(SLOW)*" if is_slow else "") + "\n"
                f"• *Cost:* `${response_payload.get('cost_usd', 0.0):.6f}`\n"
            )
            if has_guardrails:
                text += f"• *Guardrail Warnings:* {response_payload.get('guardrail_warnings')}\n"

            self._dispatch_slack(text)

        return response_payload

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        if not self.webhook_url:
            return

        text = (
            f":rotating_light: *OptiLLM Gateway Error*\n"
            f"• *Type:* `{error.__class__.__name__}`\n"
            f"• *Detail:* `{str(error)}`\n"
            f"• *Stage:* `{context.get('stage', 'runtime')}`"
        )
        self._dispatch_slack(text)

    def _dispatch_slack(self, text: str):
        def _send():
            try:
                body = json.dumps({"text": text}).encode("utf-8")
                req = urllib.request.Request(
                    self.webhook_url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5.0):
                    pass
            except Exception as e:
                logger.warning("Failed to post Slack notification: %s", e)

        threading.Thread(target=_send, daemon=True).start()
