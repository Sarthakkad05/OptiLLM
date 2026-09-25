"""
Callback Manager — Pre/Post Call Event System
Fires async callbacks on request success/failure for observability integrations.

Built-in integrations:
  - Langfuse (LLM observability)
  - Slack (budget alerts)

Custom callbacks can be registered via CallbackManager.register().
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("optillm.core.callbacks")


# ── Base Callback ─────────────────────────────────────────────────────────────


class BaseCallback:
    """Abstract base class for OptiLLM observability callbacks."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    async def async_log_success(self, payload: Dict[str, Any]) -> None:
        """Called after a successful LLM completion. Override to implement."""
        pass

    async def async_log_failure(self, payload: Dict[str, Any]) -> None:
        """Called when a request fails. Override to implement."""
        pass


# ── Callback Manager ──────────────────────────────────────────────────────────


class CallbackManager:
    """
    Manages async callback hooks for request observability.

    Callbacks are fired as background asyncio tasks — they never block request latency.
    """

    def __init__(self):
        self._callbacks: List[BaseCallback] = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_default_callbacks()

    def _load_default_callbacks(self):
        """Load built-in callbacks based on settings."""
        from app.core.config import settings

        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            try:
                from app.callbacks.langfuse_callback import LangfuseCallback
                self.register(LangfuseCallback())
                logger.info("Langfuse callback registered.")
            except Exception as exc:
                logger.warning("Failed to load Langfuse callback: %s", exc)

        if settings.SLACK_WEBHOOK_URL:
            try:
                from app.callbacks.slack_callback import SlackCallback
                self.register(SlackCallback())
                logger.info("Slack callback registered.")
            except Exception as exc:
                logger.warning("Failed to load Slack callback: %s", exc)

    def register(self, callback: BaseCallback):
        """Register a callback instance."""
        self._callbacks.append(callback)
        logger.info("Registered callback: %s", callback.name)

    def list_callbacks(self) -> List[Dict[str, Any]]:
        """Returns registered callback names and status."""
        self._ensure_initialized()
        return [
            {"name": cb.name, "status": "active"}
            for cb in self._callbacks
        ]

    def _fire(self, coro):
        """Fire a coroutine as a background task without blocking."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(coro)
            else:
                loop.run_until_complete(coro)
        except Exception as exc:
            logger.warning("Callback dispatch error: %s", exc)

    async def _run_success(self, payload: Dict[str, Any]):
        for cb in self._callbacks:
            try:
                await cb.async_log_success(payload)
            except Exception as exc:
                logger.warning("Callback '%s' success error: %s", cb.name, exc)

    async def _run_failure(self, payload: Dict[str, Any]):
        for cb in self._callbacks:
            try:
                await cb.async_log_failure(payload)
            except Exception as exc:
                logger.warning("Callback '%s' failure error: %s", cb.name, exc)

    def fire_success(self, payload: Dict[str, Any]):
        """Fire all success callbacks (non-blocking)."""
        self._ensure_initialized()
        if not self._callbacks:
            return
        self._fire(self._run_success(payload))

    def fire_failure(self, payload: Dict[str, Any]):
        """Fire all failure callbacks (non-blocking)."""
        self._ensure_initialized()
        if not self._callbacks:
            return
        self._fire(self._run_failure(payload))


# Global singleton
_callback_manager: Optional[CallbackManager] = None


def get_callback_manager() -> CallbackManager:
    global _callback_manager
    if _callback_manager is None:
        _callback_manager = CallbackManager()
    return _callback_manager
