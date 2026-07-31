from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import RequestLog
from app.schemas.analytics import AnalyticsResponse
from app.services import analytics as analytics_service
from app.engine.cache import get_cache_stats, clear_cache
from app.engine.router import ROUTING_TABLE, Complexity, update_routing_config, get_routing_config

router = APIRouter()



@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics(db: Session = Depends(get_db)):
    """
    Returns the full analytics payload.

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


# ── Cache Management ───────────────────────────────────────────────────────────

@router.delete("/cache/clear", tags=["Cache"])
def clear_cache_endpoint(db: Session = Depends(get_db)):
    """
    Wipe all semantic cache entries and reset the FAISS index.
    Use this when underlying data changes and cached responses are stale.
    """
    deleted = clear_cache(db)
    return {"deleted_entries": deleted, "message": f"Cache cleared — {deleted} entries removed."}


# ── Router Configuration ───────────────────────────────────────────────────────

class RouterConfigUpdate(BaseModel):
    low_model: Optional[str] = None     # Model for LOW complexity tasks
    medium_model: Optional[str] = None  # Model for MEDIUM complexity tasks
    low_score_threshold: Optional[int] = None    # Score <= this → LOW
    medium_score_threshold: Optional[int] = None # Score <= this → MEDIUM


@router.get("/router/config", tags=["Router"])
def get_router_config():
    """Returns the current model routing configuration."""
    return get_routing_config()


@router.post("/router/config", tags=["Router"])
def update_router_config(update: RouterConfigUpdate):
    """
    Update the model routing table at runtime — no restart required.

    Example: lower the cost ceiling by routing more to gemini-2.0-flash:
        POST /api/v1/router/config
        {"low_model": "gemini-2.0-flash", "medium_model": "gemini-2.0-flash"}
    """
    changes = update.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No changes provided.")
    updated = update_routing_config(**changes)
    return {"message": "Routing config updated.", "new_config": updated}

