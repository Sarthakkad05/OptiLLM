"""
GET /api/v1/providers/status
Real-time provider health, circuit breaker state, latency, and rotation status.
"""

from typing import Any, Dict

from fastapi import APIRouter

from app.core.circuit_breaker import circuit_breaker
from app.core.config import settings
from app.providers.registry import provider_registry

router = APIRouter()


@router.get("/status", tags=["Providers"])
async def get_providers_status() -> Dict[str, Any]:
    """
    Returns real-time status, circuit breaker state (CLOSED/OPEN/HALF_OPEN),
    p50 latency metrics, and availability for all registered providers.
    """
    providers_status = []
    circuit_states = circuit_breaker.get_all_status()

    for name in provider_registry.list_providers():
        provider = provider_registry.get(name)
        cb_info = circuit_states.get(name, {})

        is_configured = provider.is_available() if provider else False
        is_healthy = cb_info.get("is_available", True) if is_configured else False

        providers_status.append(
            {
                "name": name,
                "configured": is_configured,
                "healthy": is_healthy,
                "circuit_state": cb_info.get("state", "CLOSED"),
                "consecutive_failures": cb_info.get("failure_count", 0),
                "success_count": cb_info.get("success_count", 0),
                "latency_ms": cb_info.get("latency_ms", 0.0),
            }
        )

    return {
        "routing_strategy": settings.ROUTING_STRATEGY,
        "available_providers_count": len(provider_registry.list_available_providers()),
        "providers": providers_status,
    }
