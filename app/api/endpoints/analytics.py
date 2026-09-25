from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import RequestLog
from app.db.session import get_db
from app.engine.cache import clear_cache, get_cache_stats, warm_cache
from app.engine.router import ROUTING_TABLE, get_routing_config, update_routing_config
from app.schemas.analytics import (
    AnalyticsResponse,
    LatencyPercentilesResponse,
    ProviderAnalyticsResponse,
    SavingsBreakdownResponse,
    TokenTrendsResponse,
)
from app.schemas.evaluation import QualityAnalyticsResponse
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

    total_shadow_disagreements = (
        db.query(func.count(RequestLog.id))
        .filter(RequestLog.shadow_disagreement.is_(True))
        .scalar()
        or 0
    )

    return {
        "total_routed_requests": total_routed,
        "routing_rate": round(total_routed / total_requests, 4),
        "total_savings_usd": round(float(total_routing_savings), 6),
        "routed_model_breakdown": [
            {"model": r.model_used, "count": r.count} for r in model_breakdown
        ],
        "routing_table": routing_table_info,
        "shadow_disagreement_count": total_shadow_disagreements,
        "shadow_disagreement_rate": round(total_shadow_disagreements / total_requests, 4),
    }


@router.get("/routing/shadow-disagreements", tags=["Analytics"])
def get_shadow_disagreements(
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    Lists recent requests where the shadow-mode AI router's predicted complexity
    differed from the rule-based router's decision. Shadow mode never acts on
    these — it only observes — so this is the evidence needed to decide whether
    promoting the AI router out of shadow mode would actually help.
    """
    rows = (
        db.query(RequestLog)
        .filter(RequestLog.shadow_disagreement.is_(True))
        .order_by(RequestLog.timestamp.desc())
        .limit(min(limit, 200))
        .all()
    )
    return {
        "count": len(rows),
        "disagreements": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "prompt_snippet": r.prompt_snippet,
                "model_requested": r.model_requested,
                "model_used": r.model_used,
                "ai_predicted_complexity": r.ai_predicted_complexity,
            }
            for r in rows
        ],
    }


# ── Cache Management ───────────────────────────────────────────────────────────


@router.get("/cache/info", tags=["Cache"])
def get_cache_info_endpoint(db: Session = Depends(get_db)):
    """Returns detailed cache health, configuration, and connectivity status."""
    return get_cache_stats(db)


@router.post("/cache/warm", tags=["Cache"])
def warm_cache_endpoint(
    namespace: Optional[str] = Query(None), db: Session = Depends(get_db)
):
    """
    Pre-loads DB cache entries into Redis and FAISS vector index.
    """
    count = warm_cache(db, namespace=namespace)
    return {
        "warmed_entries": count,
        "message": f"Cache warm complete — {count} entries loaded into Redis/FAISS.",
    }


@router.delete("/cache/clear", tags=["Cache"])
def clear_cache_endpoint(
    namespace: Optional[str] = Query(None), db: Session = Depends(get_db)
):
    """
    Wipe all semantic cache entries from DB, FAISS, and Redis.
    """
    deleted = clear_cache(db, namespace=namespace)
    return {
        "deleted_entries": deleted,
        "message": f"Cache cleared — {deleted} entries removed.",
    }


# ── Phase 11 Quality Analytics ────────────────────────────────────────────────


@router.get(
    "/analytics/quality",
    response_model=QualityAnalyticsResponse,
    tags=["Analytics"],
)
def get_quality_analytics_endpoint(
    start_date: Optional[str] = Query(None, description="Start date filter"),
    end_date: Optional[str] = Query(None, description="End date filter"),
    provider: Optional[str] = Query(None, description="Provider filter"),
    tag: Optional[str] = Query(None, description="Tag filter"),
    db: Session = Depends(get_db),
):
    """
    Returns quality analytics, 5-dimension averages, hallucination metrics, and model quality ratings.
    """
    start_dt = _parse_datetime(start_date)
    end_dt = _parse_datetime(end_date)
    return analytics_service.get_quality_analytics(
        db, start_date=start_dt, end_date=end_dt, provider=provider, tag=tag
    )


@router.get(
    "/analytics/cost-attribution",
    tags=["Analytics"],
    summary="Cost Attribution & Savings Waterfall",
    description="Returns detailed spend attribution by model and team, savings waterfall breakdown, provider latency percentiles, and SLA compliance.",
)
def get_cost_attribution_endpoint(
    start_date: Optional[str] = Query(None, description="Start date filter (ISO or YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO or YYYY-MM-DD)"),
    db: Session = Depends(get_db),
):
    """
    Returns cost attribution by model and team, savings waterfall, provider latency percentiles, and SLA compliance.
    """
    start_dt = _parse_datetime(start_date)
    end_dt = _parse_datetime(end_date)
    return analytics_service.get_cost_attribution(db, start_date=start_dt, end_date=end_dt)






