from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import RequestLog
from app.db.session import get_db
from app.engine.cache import clear_cache, get_cache_stats
from app.engine.router import ROUTING_TABLE, get_routing_config, update_routing_config
from app.schemas.analytics import (
    AnalyticsResponse,
    LatencyPercentilesResponse,
    ProviderAnalyticsResponse,
    SavingsBreakdownResponse,
    TokenTrendsResponse,
)
from app.services import analytics as analytics_service

router = APIRouter()


def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: '{date_str}'. Expected ISO 8601 or YYYY-MM-DD.",
            )


@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics(db: Session = Depends(get_db)):
    """
    Returns the full analytics payload.
    """
    return {
        "summary": analytics_service.get_dashboard_summary(db),
        "cost_over_time": analytics_service.get_cost_over_time(db),
        "model_distribution": analytics_service.get_model_distribution(db),
        "recent_requests": analytics_service.get_recent_requests(db),
    }


# ── Phase 4 Deep Analytics Endpoints ──────────────────────────────────────────


@router.get(
    "/analytics/latency",
    response_model=LatencyPercentilesResponse,
    tags=["Analytics"],
)
def get_latency_percentiles_endpoint(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Returns p50, p95, and p99 latency percentiles overall and breakdown per provider.
    Supports filtering by start_date, end_date, provider, and tag.
    """
    dt_start = _parse_datetime(start_date)
    dt_end = _parse_datetime(end_date)
    return analytics_service.get_latency_percentiles(
        db, start_date=dt_start, end_date=dt_end, provider=provider, tag=tag
    )


@router.get(
    "/analytics/tokens",
    response_model=TokenTrendsResponse,
    tags=["Analytics"],
)
def get_token_trends_endpoint(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Returns daily token consumption trends (input tokens, output tokens, tokens saved).
    Supports filtering by start_date, end_date, and tag.
    """
    dt_start = _parse_datetime(start_date)
    dt_end = _parse_datetime(end_date)
    return analytics_service.get_token_trends(
        db, start_date=dt_start, end_date=dt_end, tag=tag
    )


@router.get(
    "/analytics/providers",
    response_model=ProviderAnalyticsResponse,
    tags=["Analytics"],
)
def get_provider_analytics_endpoint(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Returns per-provider analytics (requests, avg latency, cache hit rate, cost, savings).
    Supports filtering by start_date and end_date.
    """
    dt_start = _parse_datetime(start_date)
    dt_end = _parse_datetime(end_date)
    return analytics_service.get_provider_analytics(
        db, start_date=dt_start, end_date=dt_end
    )


@router.get(
    "/analytics/savings",
    response_model=SavingsBreakdownResponse,
    tags=["Analytics"],
)
def get_savings_breakdown_endpoint(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Returns savings breakdown across independent dimensions (cache, compression, routing).
    Supports filtering by start_date, end_date, and tag.
    """
    dt_start = _parse_datetime(start_date)
    dt_end = _parse_datetime(end_date)
    return analytics_service.get_savings_breakdown(
        db, start_date=dt_start, end_date=dt_end, tag=tag
    )


@router.get("/cache/stats", tags=["Analytics"])
def get_cache_stats_endpoint(db: Session = Depends(get_db)):
    """Returns semantic cache health metrics."""
    return get_cache_stats(db)


@router.get("/compression/stats", tags=["Analytics"])
def get_compression_stats(db: Session = Depends(get_db)):
    """Returns aggregated compression metrics."""
    total_compressed = (
        db.query(func.count(RequestLog.id))
        .filter(RequestLog.compressed.is_(True))
        .scalar()
        or 0
    )
    total_tokens_saved = (
        db.query(func.sum(RequestLog.tokens_saved))
        .filter(RequestLog.compressed.is_(True))
        .scalar()
        or 0
    )
    total_compression_savings = (
        db.query(func.sum(RequestLog.savings_usd))
        .filter(RequestLog.compressed.is_(True), RequestLog.cache_hit.is_(False))
        .scalar()
        or 0.0
    )

    total_requests = db.query(func.count(RequestLog.id)).scalar() or 1

    return {
        "total_compressed_requests": total_compressed,
        "compression_rate": round(total_compressed / total_requests, 4),
        "total_tokens_saved": int(total_tokens_saved),
        "total_savings_usd": round(float(total_compression_savings), 6),
    }


@router.get("/routing/stats", tags=["Analytics"])
def get_routing_stats(db: Session = Depends(get_db)):
    """Returns model routing decisions and savings breakdown."""
    total_routed = (
        db.query(func.count(RequestLog.id)).filter(RequestLog.routed.is_(True)).scalar()
        or 0
    )
    total_routing_savings = (
        db.query(func.sum(RequestLog.savings_usd))
        .filter(RequestLog.routed.is_(True), RequestLog.cache_hit.is_(False))
        .scalar()
        or 0.0
    )
    total_requests = db.query(func.count(RequestLog.id)).scalar() or 1

    model_breakdown = (
        db.query(
            RequestLog.model_used,
            func.count(RequestLog.id).label("count"),
        )
        .filter(RequestLog.routed.is_(True))
        .group_by(RequestLog.model_used)
        .all()
    )

    routing_table_info = {
        level.value: model for level, (model, _) in ROUTING_TABLE.items() if model
    }

    return {
        "total_routed_requests": total_routed,
        "routing_rate": round(total_routed / total_requests, 4),
        "total_savings_usd": round(float(total_routing_savings), 6),
        "routed_model_breakdown": [
            {"model": r.model_used, "count": r.count} for r in model_breakdown
        ],
        "routing_table": routing_table_info,
    }


# ── Cache Management ───────────────────────────────────────────────────────────


@router.delete("/cache/clear", tags=["Cache"])
def clear_cache_endpoint(db: Session = Depends(get_db)):
    """
    Wipe all semantic cache entries and reset the FAISS index.
    """
    deleted = clear_cache(db)
    return {
        "deleted_entries": deleted,
        "message": f"Cache cleared — {deleted} entries removed.",
    }


# ── Router Configuration ───────────────────────────────────────────────────────


class RouterConfigUpdate(BaseModel):
    low_model: Optional[str] = None
    medium_model: Optional[str] = None
    low_score_threshold: Optional[int] = None
    medium_score_threshold: Optional[int] = None


@router.get("/router/config", tags=["Router"])
def get_router_config():
    """Returns the current model routing configuration."""
    return get_routing_config()


@router.post("/router/config", tags=["Router"])
def update_router_config(update: RouterConfigUpdate):
    """
    Update the model routing table at runtime — no restart required.
    """
    changes = update.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No changes provided.")
    updated = update_routing_config(**changes)
    return {"message": "Routing config updated.", "new_config": updated}
