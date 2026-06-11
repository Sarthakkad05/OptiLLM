"""
Analytics Service
Aggregates request logs for the dashboard and API endpoints.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
from typing import List, Dict, Any
from app.db.models import RequestLog
import logging

logger = logging.getLogger("optillm.analytics")


def get_dashboard_summary(db: Session) -> Dict[str, Any]:
    """Top-level KPI summary for the dashboard."""
    total = db.query(func.count(RequestLog.id)).scalar() or 0
    cache_hits = db.query(func.count(RequestLog.id)).filter(RequestLog.cache_hit == True).scalar() or 0  # noqa: E712
    total_cost = db.query(func.sum(RequestLog.cost_usd)).scalar() or 0.0
    total_savings = db.query(func.sum(RequestLog.savings_usd)).scalar() or 0.0
    total_tokens_saved = db.query(func.sum(RequestLog.tokens_saved)).scalar() or 0
    avg_latency = db.query(func.avg(RequestLog.latency_ms)).scalar() or 0.0
    total_tokens_in = db.query(func.sum(RequestLog.tokens_input)).scalar() or 0
    total_tokens_out = db.query(func.sum(RequestLog.tokens_output)).scalar() or 0

    cache_hit_rate = round(cache_hits / total, 4) if total > 0 else 0.0

    return {
        "total_requests": total,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "total_cost_usd": round(float(total_cost), 6),
        "total_savings_usd": round(float(total_savings), 6),
        "total_tokens_input": int(total_tokens_in),
        "total_tokens_output": int(total_tokens_out),
        "total_tokens_saved": int(total_tokens_saved),
        "avg_latency_ms": round(float(avg_latency), 2),
    }


def get_recent_requests(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch the most recent requests for the dashboard table."""
    rows = (
        db.query(RequestLog)
        .order_by(RequestLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "model_used": r.model_used,
            "provider": r.provider,
            "cache_hit": r.cache_hit,
            "compressed": r.compressed,
            "routed": r.routed,
            "tokens_input": r.tokens_input,
            "tokens_output": r.tokens_output,
            "tokens_saved": r.tokens_saved,
            "cost_usd": r.cost_usd,
            "savings_usd": r.savings_usd,
            "latency_ms": r.latency_ms,
            "prompt_snippet": r.prompt_snippet,
        }
        for r in rows
    ]


def get_cost_over_time(db: Session) -> List[Dict[str, Any]]:
    """Daily aggregated cost and savings — used for the line chart."""
    rows = (
        db.query(
            cast(RequestLog.timestamp, Date).label("date"),
            func.sum(RequestLog.cost_usd).label("cost"),
            func.sum(RequestLog.savings_usd).label("savings"),
            func.count(RequestLog.id).label("requests"),
        )
        .group_by(cast(RequestLog.timestamp, Date))
        .order_by(cast(RequestLog.timestamp, Date))
        .all()
    )
    return [
        {
            "date": str(r.date),
            "cost_usd": round(float(r.cost), 6),
            "savings_usd": round(float(r.savings), 6),
            "requests": r.requests,
        }
        for r in rows
    ]


def get_model_distribution(db: Session) -> List[Dict[str, Any]]:
    """Per-model request counts — used for the pie/bar chart."""
    rows = (
        db.query(
            RequestLog.model_used,
            func.count(RequestLog.id).label("count"),
        )
        .group_by(RequestLog.model_used)
        .all()
    )
    return [{"model": r.model_used, "requests": r.count} for r in rows]
