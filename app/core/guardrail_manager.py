"""
GuardrailManager — Pre/Post Call Hook Registry
Manages a pipeline of guardrail functions that run before and after LLM calls.

Built-in hooks:
  - pii_redact        : Detects and redacts PII from user messages (pre-call)
  - content_safety    : Keyword blocklist check (pre-call)
  - prompt_injection  : Detects classic prompt injection patterns (pre-call)

Custom hooks can be registered via register_pre_call() / register_post_call().
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.config import settings
from app.security.pii import get_pii_detector

logger = logging.getLogger("optillm.core.guardrails")


# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class GuardrailResult:
    """Result returned by each guardrail hook."""

    allowed: bool = True
    modified_messages: Optional[List[Dict]] = None  # None = unchanged
    modified_response: Optional[str] = None  # None = unchanged
    warnings: List[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailHook:
    name: str
    fn: Callable
    mode: str  # "pre_call" | "post_call"
    enabled: bool = True
    block_on_trigger: bool = False  # If True, blocks request when triggered


# ── Built-in Guardrail Functions ──────────────────────────────────────────────


def _pii_redact_hook(
    messages: List[Dict], response: Optional[str] = None
) -> GuardrailResult:
    """Scans user messages for PII and redacts them in-place."""
    detector = get_pii_detector()
    modified = False
    warnings = []
    new_messages = []

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            new_messages.append(msg)
            continue

        result = detector.redact(content)
        if result["pii_found"]:
            types = result["detected_types"]
            warnings.append(f"PII detected and redacted: {types}")
            logger.warning(
                "PII detected in message (role=%s): %s", msg.get("role"), types
            )
            new_msg = dict(msg)
            new_msg["content"] = result["redacted_text"]
            new_messages.append(new_msg)
            modified = True
        else:
            new_messages.append(msg)

    return GuardrailResult(
        allowed=True,
        modified_messages=new_messages if modified else None,
        warnings=warnings,
        metadata={"pii_detected": modified},
    )


# Content safety keyword blocklist
_BLOCKED_KEYWORDS = [
    "jailbreak", "ignore previous instructions", "disregard all prior",
    "forget your instructions", "you are now", "pretend you are",
    "act as if you have no restrictions", "dan mode", "developer mode",
]

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I),
    re.compile(r"you\s+are\s+now\s+\w+", re.I),
    re.compile(r"system\s*:\s*you\s+are", re.I),
    re.compile(r"</?(system|user|assistant)>", re.I),
    re.compile(r"\[INST\]|\[/INST\]", re.I),
]


def _content_safety_hook(
    messages: List[Dict], response: Optional[str] = None
) -> GuardrailResult:
    """Keyword-based content safety check on user messages."""
    user_text = " ".join(
        m.get("content", "") for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    ).lower()

    for keyword in _BLOCKED_KEYWORDS:
        if keyword in user_text:
            logger.warning("Content safety block: keyword '%s' detected", keyword)
            return GuardrailResult(
                allowed=False,
                blocked_reason=f"Request blocked by content safety policy (keyword: '{keyword}').",
                warnings=[f"Blocked keyword: {keyword}"],
            )

    return GuardrailResult(allowed=True)


def _prompt_injection_hook(
    messages: List[Dict], response: Optional[str] = None
) -> GuardrailResult:
    """Detects common prompt injection attack patterns."""
    user_text = " ".join(
        m.get("content", "") for m in messages
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(user_text):
            logger.warning("Prompt injection detected: pattern '%s'", pattern.pattern)
            return GuardrailResult(
                allowed=False,
                blocked_reason="Request blocked: prompt injection pattern detected.",
                warnings=[f"Injection pattern matched: {pattern.pattern}"],
            )

    return GuardrailResult(allowed=True)


# ── GuardrailManager ──────────────────────────────────────────────────────────


class GuardrailManager:
    """
    Manages ordered lists of pre_call and post_call guardrail hooks.
    Hooks are executed in registration order. First blocking result wins.
    """

    def __init__(self):
        self._hooks: List[GuardrailHook] = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True

        # Register built-in hooks based on settings
        if settings.GUARDRAIL_PII_ENABLED:
            self.register(GuardrailHook(
                name="pii_redact",
                fn=_pii_redact_hook,
                mode="pre_call",
                enabled=True,
                block_on_trigger=False,  # Redact but don't block
            ))

        if settings.GUARDRAIL_CONTENT_SAFETY_ENABLED:
            self.register(GuardrailHook(
                name="content_safety",
                fn=_content_safety_hook,
                mode="pre_call",
                enabled=True,
                block_on_trigger=True,
            ))

        if settings.GUARDRAIL_PROMPT_INJECTION_ENABLED:
            self.register(GuardrailHook(
                name="prompt_injection",
                fn=_prompt_injection_hook,
                mode="pre_call",
                enabled=True,
                block_on_trigger=True,
            ))

    def register(self, hook: GuardrailHook):
        """Register a guardrail hook."""
        self._hooks.append(hook)
        logger.info(
            "Registered guardrail hook: name=%s mode=%s block=%s",
            hook.name, hook.mode, hook.block_on_trigger,
        )

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a hook by name. Returns True if found."""
        for hook in self._hooks:
            if hook.name == name:
                hook.enabled = enabled
                logger.info("Guardrail '%s' set to enabled=%s", name, enabled)
                return True
        return False

    def list_hooks(self) -> List[Dict[str, Any]]:
        """Returns list of all registered hooks with their status."""
        self._ensure_initialized()
        return [
            {
                "name": h.name,
                "mode": h.mode,
                "enabled": h.enabled,
                "block_on_trigger": h.block_on_trigger,
            }
            for h in self._hooks
        ]

    def run_pre_call(
        self, messages: List[Dict]
    ) -> Tuple[bool, List[Dict], List[str], Optional[str]]:
        """
        Run all enabled pre_call hooks sequentially.

        Returns:
            (allowed, messages, warnings, blocked_reason)
        """
        self._ensure_initialized()
        if not settings.GUARDRAILS_ENABLED:
            return True, messages, [], None

        current_messages = messages
        all_warnings: List[str] = []

        for hook in self._hooks:
            if hook.mode != "pre_call" or not hook.enabled:
                continue
            try:
                result = hook.fn(current_messages)
                all_warnings.extend(result.warnings)

                if result.modified_messages is not None:
                    current_messages = result.modified_messages

                if not result.allowed:
                    logger.warning(
                        "Guardrail '%s' blocked request: %s",
                        hook.name,
                        result.blocked_reason,
                    )
                    return False, current_messages, all_warnings, result.blocked_reason

            except Exception as exc:
                logger.error("Guardrail hook '%s' raised: %s", hook.name, exc, exc_info=True)
                # Never block on guardrail errors — fail open

        return True, current_messages, all_warnings, None

    def run_post_call(
        self, response: str, messages: List[Dict]
    ) -> Tuple[str, List[str]]:
        """
        Run all enabled post_call hooks.

        Returns:
            (response, warnings)
        """
        self._ensure_initialized()
        if not settings.GUARDRAILS_ENABLED:
            return response, []

        current_response = response
        all_warnings: List[str] = []

        for hook in self._hooks:
            if hook.mode != "post_call" or not hook.enabled:
                continue
            try:
                result = hook.fn(messages, response=current_response)
                all_warnings.extend(result.warnings)
                if result.modified_response is not None:
                    current_response = result.modified_response
            except Exception as exc:
                logger.error("Post-call guardrail '%s' raised: %s", hook.name, exc, exc_info=True)

        return current_response, all_warnings

    def test_prompt(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        Dry-run all pre_call hooks on a prompt and return a full report.
        Used by POST /api/v1/guardrails/test.
        """
        self._ensure_initialized()
        results = []
        for hook in self._hooks:
            if hook.mode != "pre_call":
                continue
            try:
                result = hook.fn(messages)
                results.append({
                    "hook": hook.name,
                    "enabled": hook.enabled,
                    "allowed": result.allowed,
                    "warnings": result.warnings,
                    "blocked_reason": result.blocked_reason,
                    "metadata": result.metadata,
                })
            except Exception as exc:
                results.append({
                    "hook": hook.name,
                    "enabled": hook.enabled,
                    "allowed": True,
                    "warnings": [],
                    "error": str(exc),
                })

        overall_blocked = any(
            not r["allowed"] for r in results if r.get("enabled", True)
        )
        return {
            "overall_allowed": not overall_blocked,
            "hooks": results,
        }


# Global singleton
_guardrail_manager: Optional[GuardrailManager] = None


def get_guardrail_manager() -> GuardrailManager:
    global _guardrail_manager
    if _guardrail_manager is None:
        _guardrail_manager = GuardrailManager()
    return _guardrail_manager
