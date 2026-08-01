"""
Provider Health Monitor Background Task.
Periodically probes registered LLM providers, records latency and health metrics,
and maintains real-time circuit breaker status.
"""

import asyncio
import logging
from typing import Any, Dict

from app.core.circuit_breaker import circuit_breaker
from app.core.config import settings
from app.providers.registry import provider_registry

logger = logging.getLogger("optillm.provider_health")

_health_task: asyncio.Task | None = None


async def probe_provider_health(provider_name: str) -> Dict[str, Any]:
    """Probes a single provider's availability and measures health/latency."""
    provider = provider_registry.get(provider_name)
    if not provider:
        return {"name": provider_name, "available": False, "status": "not_registered"}

    if not provider.is_available():
        return {
            "name": provider_name,
            "available": False,
            "status": "missing_credentials",
        }

    # Record current circuit state
    cb_state = circuit_breaker.get_circuit(provider_name)
    return {
        "name": provider_name,
        "available": True,
        "circuit_state": cb_state.state.value,
        "latency_ms": cb_state.latency_ms,
        "consecutive_failures": cb_state.failure_count,
        "total_successes": cb_state.success_count,
    }


async def run_health_check_loop():
    """Background task loop periodically checking provider health status."""
    logger.info(
        "🚀 Starting provider health check monitor (interval: %ds)...",
        settings.HEALTH_CHECK_INTERVAL_SECONDS,
    )
    while True:
        try:
            for name in provider_registry.list_providers():
                await probe_provider_health(name)
        except asyncio.CancelledError:
            logger.info("🛑 Provider health check loop cancelled.")
            break
        except Exception as exc:
            logger.error("Error in provider health check loop: %s", exc)

        await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL_SECONDS)


def start_health_check_task():
    """Starts the provider health check background task."""
    global _health_task
    if _health_task is None or _health_task.done():
        _health_task = asyncio.create_task(run_health_check_loop())


def stop_health_check_task():
    """Stops the provider health check background task."""
    global _health_task
    if _health_task and not _health_task.done():
        _health_task.cancel()
        _health_task = None
