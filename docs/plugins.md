# OptiLLM Plugins Architecture & Extension Guide

OptiLLM provides an enterprise plugin framework that enables engineering teams to inject custom logic into the request/response lifecycle without touching the core gateway code.

Plugins can inspect, enrich, validate, or mutate prompts before provider dispatch, transform completion responses, enforce security policies, or emit asynchronous telemetry to third-party endpoints.

---

## 1. Plugin Lifecycle Overview

Every request passes through the following pipeline:

```
Inbound HTTP Request (/v1/chat/completions)
   │
   ▼
[1. Plugin pre_process()] ──────── Enrich prompt, inject headers, validate PII
   │
   ▼
[2. Cache & Optimization] ──────── Semantic cache lookup, TF-IDF compression, Model routing
   │
   ▼
[3. Provider Call] ────────────── OpenAI, Anthropic, Gemini, Ollama, etc.
   │
   ▼
[4. Plugin post_process()] ────── Compliance signatures, auditing, Slack/Webhook dispatches
   │
   ▼
Outbound Response to Client
```

If an unhandled exception occurs at any point, registered plugins have their `on_error()` method invoked for graceful degradation and alert delivery.

---

## 2. Base Plugin Interface

All plugins inherit from `BasePlugin` (`app/plugins/base.py`):

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BasePlugin(ABC):
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called once when the plugin is registered."""
        pass

    @abstractmethod
    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoked before LLM provider request dispatch.
        Can modify messages, parameters, or inject system instructions.
        """
        return request_payload

    @abstractmethod
    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoked after provider completion is returned.
        Can transform response content or append enterprise metadata.
        """
        return response_payload

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Invoked if an error occurs during gateway processing."""
        pass
```

---

## 3. Included Built-in Plugins

OptiLLM ships with four production-ready plugins:

### 1. PIIGuardPlugin (`app/plugins/reference_plugin.py`)
Scans inbound user prompts using Microsoft Presidio & spaCy and automatically redacts emails, SSNs, credit cards, and phone numbers before the prompt leaves your network.

### 2. CustomHeaderPlugin (`app/plugins/reference_plugin.py`)
Appends enterprise compliance and policy verification headers (`enterprise_compliance`) to every outbound completion.

### 3. WebhookPlugin (`app/plugins/webhook_plugin.py`)
Dispatches completion and error events to any HTTP webhook asynchronously in a detached background thread, including HMAC-SHA256 signature verification (`X-OptiLLM-Signature`).

```python
from app.plugins.webhook_plugin import WebhookPlugin
from app.plugins.registry import get_plugin_registry

webhook = WebhookPlugin(
    name="enterprise_audit_webhook",
    webhook_url="https://audit.company.internal/llm-events",
    secret_token="super-secret-hmac-key"
)
get_plugin_registry().register(webhook)
```

### 4. SlackNotificationPlugin (`app/plugins/slack_notification_plugin.py`)
Alerts engineering teams on Slack / Discord / Microsoft Teams whenever latency exceeds a SLA threshold (e.g., > 3,000 ms), guardrails are triggered, or provider outages occur.

---

## 4. Authoring a Custom Plugin

Here is an example of a custom plugin that appends a corporate copyright notice and tracks internal project IDs:

```python
from app.plugins.base import BasePlugin
from typing import Any, Dict, Optional

class CorporateNoticePlugin(BasePlugin):
    def __init__(self, name: str = "corporate_notice", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)
        self.notice_text = ""

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self.notice_text = config.get("notice", "\n\n© 2026 Acme Corp. All rights reserved.")

    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Prepend corporate disclaimer to system prompt
        messages = request_payload.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += " Always adhere to Acme code of conduct."
        return request_payload

    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Append corporate notice to content
        if "content" in response_payload and isinstance(response_payload["content"], str):
            response_payload["content"] += self.notice_text
        return response_payload
```

---

## 5. Runtime Registration & Dynamic Toggling

Plugins can be registered at startup in your custom lifecycle or dynamically enabled/disabled via the `PluginRegistry`:

```python
from app.plugins.registry import get_plugin_registry

registry = get_plugin_registry()

# Register
registry.register(CorporateNoticePlugin(), config={"notice": "Acme Protected"})

# Disable on the fly without server restart
registry.toggle("corporate_notice", enabled=False)

# List active plugins
active = registry.list_plugins()
```
