"""
Live Dashboard Endpoint.
Returns a single JSON snapshot of all live metrics needed by the Admin UI.
Designed to be polled every 2-5 seconds from the frontend.
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import RequestLog, CacheEntry
from app.db.session import get_db
from app.providers.registry import provider_registry

router = APIRouter(tags=["Dashboard"])


@router.get(
    "/dashboard/live",
    summary="Live Dashboard Metrics",
    description=(
        "Returns a consolidated snapshot of all live metrics for the Admin UI: "
        "active requests, today's cost, cache hit rate, provider health, and more. "
        "Poll every 2-5 seconds."
    ),
)
def live_dashboard(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Returns live metrics snapshot for Admin UI polling."""
    now = time.time()
    now_int = int(now)
    one_min_ago = now_int - 60
    today_midnight = now_int - (now_int % 86400)
    one_hour_ago = now_int - 3600

    # --- Requests in last 60s ---
    try:
        recent_count = (
            db.query(func.count(RequestLog.id))
            .filter(RequestLog.created_at >= one_min_ago)
            .scalar() or 0
        )
    except Exception:
        recent_count = 0

    # --- Today's cost ---
    try:
        today_cost = (
            db.query(func.sum(RequestLog.cost_usd))
            .filter(RequestLog.created_at >= today_midnight)
            .scalar() or 0.0
        )
    except Exception:
        today_cost = 0.0

    # --- Cache hit rate (last hour) ---
    try:
        total_requests_hour = (
            db.query(func.count(RequestLog.id))
            .filter(RequestLog.created_at >= one_hour_ago)
            .scalar() or 0
        )
        cache_hits_hour = (
            db.query(func.count(RequestLog.id))
            .filter(
                RequestLog.created_at >= one_hour_ago,
                RequestLog.cache_hit == True,
            )
            .scalar() or 0
        )
        cache_hit_rate = (
            round((cache_hits_hour / total_requests_hour) * 100, 1)
            if total_requests_hour > 0
            else 0.0
        )
    except Exception:
        cache_hit_rate = 0.0
        total_requests_hour = 0

    # --- Today's token usage ---
    try:
        today_tokens = (
            db.query(
                func.sum(RequestLog.tokens_input + RequestLog.tokens_output)
            )
            .filter(RequestLog.created_at >= today_midnight)
            .scalar() or 0
        )
    except Exception:
        today_tokens = 0

    # --- Today's savings ---
    try:
        today_savings = (
            db.query(func.sum(RequestLog.savings_usd))
            .filter(RequestLog.created_at >= today_midnight)
            .scalar() or 0.0
        )
    except Exception:
        today_savings = 0.0

    # --- Provider health ---
    provider_health = []
    for name in provider_registry.list_providers():
        provider = provider_registry.get(name)
        provider_health.append({
            "name": name,
            "available": provider.is_available() if provider else False,
        })

    # --- Guardrail status ---
    try:
        from app.core.guardrail_manager import get_guardrail_manager
        guardrail_hooks = get_guardrail_manager().list_hooks()
    except Exception:
        guardrail_hooks = []

    # --- Callback status ---
    try:
        from app.core.callback_manager import get_callback_manager
        callbacks = get_callback_manager().list_callbacks()
    except Exception:
        callbacks = []

    # --- Today's savings breakdown ---
    try:
        cache_sav_today = (
            db.query(func.sum(RequestLog.savings_usd))
            .filter(RequestLog.created_at >= today_midnight, RequestLog.cache_hit == True)
            .scalar() or 0.0
        )
        comp_sav_today = (
            db.query(func.sum(RequestLog.savings_usd))
            .filter(RequestLog.created_at >= today_midnight, RequestLog.cache_hit == False, RequestLog.compressed == True)
            .scalar() or 0.0
        )
        route_sav_today = (
            db.query(func.sum(RequestLog.savings_usd))
            .filter(RequestLog.created_at >= today_midnight, RequestLog.cache_hit == False, RequestLog.compressed == False, RequestLog.routed == True)
            .scalar() or 0.0
        )
    except Exception:
        cache_sav_today, comp_sav_today, route_sav_today = 0.0, 0.0, 0.0

    # --- SLA Compliance ---
    try:
        from app.services.sla_manager import get_sla_manager
        sla_summary = get_sla_manager().get_sla_report(db)
    except Exception:
        sla_summary = {"compliance_rate": 100.0, "status": "compliant"}

    return {
        "timestamp": now_int,
        "requests": {
            "last_60s": recent_count,
            "last_hour": total_requests_hour,
        },
        "cost": {
            "today_usd": round(today_cost, 6),
            "savings_today_usd": round(today_savings, 6),
        },
        "tokens": {
            "today_total": today_tokens,
        },
        "cache": {
            "hit_rate_percent": cache_hit_rate,
            "hits_last_hour": cache_hit_rate,
        },
        "savings_waterfall": {
            "cache_usd": round(cache_sav_today, 6),
            "compression_usd": round(comp_sav_today, 6),
            "routing_usd": round(route_sav_today, 6),
            "total_usd": round(today_savings, 6),
        },
        "sla": {
            "compliance_rate": sla_summary.get("compliance_rate", 100.0),
            "status": sla_summary.get("status", "compliant"),
            "target_ms": sla_summary.get("sla_target_ms", 500.0),
            "p95_ms": sla_summary.get("p95_latency_ms", 0.0),
        },
        "providers": provider_health,
        "guardrails": guardrail_hooks,
        "callbacks": callbacks,
    }
