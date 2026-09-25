"""
Unit and integration tests for OptiLLM Plugin Architecture.
Tests BasePlugin, PluginRegistry, WebhookPlugin, SlackNotificationPlugin,
and dynamic hook execution during gateway requests.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.plugins.base import BasePlugin
from app.plugins.registry import PluginRegistry, get_plugin_registry
from app.plugins.reference_plugin import PIIGuardPlugin, CustomHeaderPlugin
from app.plugins.webhook_plugin import WebhookPlugin
from app.plugins.slack_notification_plugin import SlackNotificationPlugin


class MockCustomPlugin(BasePlugin):
    def initialize(self, config=None):
        self.config = config or {}

    def pre_process(self, request_payload):
        msgs = request_payload.get("messages", [])
        msgs.append({"role": "system", "content": "injected-system-prompt"})
        request_payload["messages"] = msgs
        return request_payload

    def post_process(self, response_payload):
        response_payload["custom_flag"] = True
        return response_payload


def test_plugin_registry_lifecycle():
    registry = PluginRegistry()
    plugin = MockCustomPlugin(name="test_plugin", enabled=True)
    registry.register(plugin, config={"key": "value"})

    plugins = registry.list_plugins()
    assert len(plugins) == 1
    assert plugins[0]["name"] == "test_plugin"
    assert plugins[0]["enabled"] is True

    # Pre-process hook modifies payload
    req = {"messages": [{"role": "user", "content": "hi"}]}
    processed = registry.run_pre_process(req)
    assert len(processed["messages"]) == 2
    assert processed["messages"][-1]["content"] == "injected-system-prompt"

    # Post-process hook modifies payload
    resp = {"id": "123", "model": "gpt-4o"}
    processed_resp = registry.run_post_process(resp)
    assert processed_resp.get("custom_flag") is True

    # Toggle off
    assert registry.toggle("test_plugin", False) is True
    assert registry.list_plugins()[0]["enabled"] is False

    # Disabled plugin does not modify payload
    fresh_req = {"messages": [{"role": "user", "content": "hello"}]}
    unmodified = registry.run_pre_process(fresh_req)
    assert len(unmodified["messages"]) == 1


def test_pii_guard_plugin():
    plugin = PIIGuardPlugin(name="pii_test", enabled=True)
    req = {
        "messages": [
            {"role": "user", "content": "Contact me at sarthak@example.com immediately."}
        ]
    }
    redacted = plugin.pre_process(req)
    assert "[REDACTED_EMAIL]" in redacted["messages"][0]["content"]
    assert "sarthak@example.com" not in redacted["messages"][0]["content"]


def test_custom_header_plugin():
    plugin = CustomHeaderPlugin(name="header_test", enabled=True)
    resp = {"id": "resp-1", "content": "Hello world"}
    res = plugin.post_process(resp)
    assert "enterprise_compliance" in res
    assert res["enterprise_compliance"]["verified"] is True


def test_webhook_plugin_dispatch():
    plugin = WebhookPlugin(
        name="wh_test",
        enabled=True,
        webhook_url="http://mock-webhook.internal/events",
        secret_token="secret-123"
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        payload = {
            "id": "chatcmpl-test",
            "model": "gpt-4o",
            "latency_ms": 45,
            "cost_usd": 0.0015,
            "cache_hit": False,
        }
        res = plugin.post_process(payload)
        assert res["id"] == "chatcmpl-test"

        # Also test on_error
        plugin.on_error(ValueError("simulated error"), {"stage": "pre_process"})


def test_slack_notification_plugin():
    plugin = SlackNotificationPlugin(
        name="slack_test",
        enabled=True,
        webhook_url="https://hooks.slack.com/services/test/mock",
        latency_threshold_ms=1000,
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        # Should not alert for fast call
        fast_resp = {"id": "1", "model": "gpt-4o", "latency_ms": 100}
        plugin.post_process(fast_resp)

        # Should alert for slow call (exceeds threshold)
        slow_resp = {"id": "2", "model": "gpt-4o", "latency_ms": 1500}
        plugin.post_process(slow_resp)

        # Should alert on error
        plugin.on_error(RuntimeError("Out of quota"), {"stage": "provider_call"})
