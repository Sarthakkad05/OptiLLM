from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import RequestLog
from app.schemas.analytics import AnalyticsResponse
from app.services import analytics as analytics_service
from app.engine.cache import get_cache_stats

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


