"""
Reference Plugin Implementations — PIIGuardPlugin & CustomHeaderPlugin
Working reference plugins demonstrating enterprise pre/post-processing hooks.
"""

from typing import Any, Dict, Optional
from app.plugins.base import BasePlugin
from app.security.pii import get_pii_detector


class PIIGuardPlugin(BasePlugin):
    """
    Reference plugin: Scans request prompt messages for sensitive PII data
    and automatically redacts them prior to provider dispatch.
    """

    def __init__(self, name: str = "pii_guard", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        detector = get_pii_detector()
        messages = request_payload.get("messages", [])
        if not messages:
            return request_payload

        redacted_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                res = detector.redact(content)
                new_msg = dict(msg)
                new_msg["content"] = res["redacted_text"]
                redacted_messages.append(new_msg)
            else:
                redacted_messages.append(msg)

        request_payload["messages"] = redacted_messages
        return request_payload

    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        return response_payload


class CustomHeaderPlugin(BasePlugin):
    """
    Reference plugin: Appends enterprise compliance signature to completion output.
    """

    def __init__(self, name: str = "custom_header", enabled: bool = True):
        super().__init__(name=name, enabled=enabled)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return request_payload

    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        if "enterprise_compliance" not in response_payload:
            response_payload["enterprise_compliance"] = {
                "verified": True,
                "plugin_sig": "OptiLLM-Enterprise-v14",
            }
        return response_payload
