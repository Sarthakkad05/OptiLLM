from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import RequestLog
from app.schemas.analytics import AnalyticsResponse
from app.services import analytics as analytics_service
from app.engine.cache import get_cache_stats
from app.engine.router import ROUTING_TABLE, Complexity

router = APIRouter()


@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics(db: Session = Depends(get_db)):
    """
    Returns full analytics payload for the OptiLLM dashboard.

    Includes:
    - KPI summary (total requests, cache hit rate, cost saved, etc.)
    - Daily cost over time
    - Model usage distribution
    - Recent 50 requests
    """
    return {
        "summary": analytics_service.get_dashboard_summary(db),
        "cost_over_time": analytics_service.get_cost_over_time(db),
        "model_distribution": analytics_service.get_model_distribution(db),
        "recent_requests": analytics_service.get_recent_requests(db),
    }


@router.get("/cache/stats", tags=["Analytics"])
def get_cache_stats_endpoint(db: Session = Depends(get_db)):
    """Returns semantic cache health metrics."""
    return get_cache_stats(db)


@router.get("/compression/stats", tags=["Analytics"])
def get_compression_stats(db: Session = Depends(get_db)):
    """Returns aggregated compression metrics."""
    total_compressed = (
        db.query(func.count(RequestLog.id))
        .filter(RequestLog.compressed == True)  # noqa: E712
        .scalar() or 0
    )
    total_tokens_saved = (
        db.query(func.sum(RequestLog.tokens_saved))
        .filter(RequestLog.compressed == True)  # noqa: E712
        .scalar() or 0
    )
    total_compression_savings = (
        db.query(func.sum(RequestLog.savings_usd))
        .filter(RequestLog.compressed == True, RequestLog.cache_hit == False)  # noqa: E712
        .scalar() or 0.0
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
        db.query(func.count(RequestLog.id))
        .filter(RequestLog.routed == True)  # noqa: E712
        .scalar() or 0
    )
    total_routing_savings = (
        db.query(func.sum(RequestLog.savings_usd))
        .filter(RequestLog.routed == True, RequestLog.cache_hit == False)  # noqa: E712
        .scalar() or 0.0
    )
    total_requests = db.query(func.count(RequestLog.id)).scalar() or 1

    # Per-model breakdown of routed requests
    model_breakdown = (
        db.query(
            RequestLog.model_used,
            func.count(RequestLog.id).label("count"),
        )
        .filter(RequestLog.routed == True)  # noqa: E712
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
