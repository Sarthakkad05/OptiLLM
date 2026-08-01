"""
Analytics Service.
Aggregates request logs for the dashboard and Phase 4 deep analytics endpoints.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import RequestLog

logger = logging.getLogger("optillm.analytics")


def _apply_filters(
    query,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    provider: Optional[str] = None,
    tag: Optional[str] = None,
):
    """Utility helper applying date, provider, and tag filters to SQLAlchemy query."""
    if start_date:
        query = query.filter(RequestLog.timestamp >= start_date)
    if end_date:
        query = query.filter(RequestLog.timestamp <= end_date)
    if provider:
        query = query.filter(RequestLog.provider == provider)
    if tag:
        query = query.filter(RequestLog.tag == tag)
    return query


def get_dashboard_summary(db: Session) -> Dict[str, Any]:
    """Top-level KPI summary for the dashboard."""
    total = db.query(func.count(RequestLog.id)).scalar() or 0
    cache_hits = (
        db.query(func.count(RequestLog.id))
        .filter(RequestLog.cache_hit.is_(True))
        .scalar()
        or 0
    )

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
    rows = db.query(RequestLog).order_by(RequestLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "model_requested": r.model_requested,
            "model_used": r.model_used,
            "provider": r.provider,
            "tag": r.tag,
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
    date_trunc = func.date(RequestLog.timestamp)
    rows = (
        db.query(
            date_trunc.label("date"),
            func.sum(RequestLog.cost_usd).label("cost"),
            func.sum(RequestLog.savings_usd).label("savings"),
            func.count(RequestLog.id).label("requests"),
        )
        .group_by(date_trunc)
        .order_by(date_trunc)
        .all()
    )
    return [
        {
            "date": str(r.date),
            "cost_usd": round(float(r.cost or 0), 6),
            "savings_usd": round(float(r.savings or 0), 6),
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


# ── Phase 4 Deep Analytics Functions ──────────────────────────────────────────


def get_latency_percentiles(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    provider: Optional[str] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Computes p50, p95, p99 latency percentiles overall and breakdown per provider."""
    query = db.query(RequestLog.latency_ms, RequestLog.provider)
    query = _apply_filters(query, start_date, end_date, provider, tag)
    rows = query.all()

    if not rows:
        return {
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "total_requests": 0,
            "per_provider": {},
        }

    all_latencies = [r.latency_ms for r in rows]
    p50 = float(np.percentile(all_latencies, 50))
    p95 = float(np.percentile(all_latencies, 95))
    p99 = float(np.percentile(all_latencies, 99))

    # Per-provider grouping
    per_provider_latencies: Dict[str, List[int]] = {}
    for lat, prov in rows:
        if prov not in per_provider_latencies:
            per_provider_latencies[prov] = []
        per_provider_latencies[prov].append(lat)

    per_provider_stats = {}
    for prov, lat_list in per_provider_latencies.items():
        per_provider_stats[prov] = {
            "p50_ms": round(float(np.percentile(lat_list, 50)), 2),
            "p95_ms": round(float(np.percentile(lat_list, 95)), 2),
            "p99_ms": round(float(np.percentile(lat_list, 99)), 2),
            "count": len(lat_list),
        }

    return {
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "total_requests": len(rows),
        "per_provider": per_provider_stats,
    }


def get_token_trends(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Daily token consumption time series (tokens_input, tokens_output, tokens_saved)."""
    date_trunc = func.date(RequestLog.timestamp)
    query = db.query(
        date_trunc.label("date"),
        func.sum(RequestLog.tokens_input).label("input"),
        func.sum(RequestLog.tokens_output).label("output"),
        func.sum(RequestLog.tokens_saved).label("saved"),
    )
    query = _apply_filters(query, start_date, end_date, tag=tag)
    rows = query.group_by(date_trunc).order_by(date_trunc).all()

    trends = [
        {
            "date": str(r.date),
            "tokens_input": int(r.input or 0),
            "tokens_output": int(r.output or 0),
            "tokens_saved": int(r.saved or 0),
        }
        for r in rows
    ]

    return {"trends": trends}


def get_provider_analytics(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Per-provider metrics: request counts, avg latency, cache hit rate, cost, and savings."""
    query = db.query(
        RequestLog.provider,
        func.count(RequestLog.id).label("requests"),
        func.avg(RequestLog.latency_ms).label("avg_latency"),
        func.sum(RequestLog.cost_usd).label("cost"),
        func.sum(RequestLog.savings_usd).label("savings"),
    )
    query = _apply_filters(query, start_date, end_date)
    rows = query.group_by(RequestLog.provider).all()

    providers_analytics = []
    for r in rows:
        # Cache hit rate per provider
        cache_hits = (
            db.query(func.count(RequestLog.id))
            .filter(RequestLog.provider == r.provider)
            .filter(RequestLog.cache_hit.is_(True))
            .scalar()
            or 0
        )
        hit_rate = round(cache_hits / r.requests, 4) if r.requests > 0 else 0.0

        providers_analytics.append(
            {
                "provider": r.provider,
                "request_count": r.requests,
                "avg_latency_ms": round(float(r.avg_latency or 0), 2),
                "cache_hit_rate": hit_rate,
                "cost_usd": round(float(r.cost or 0.0), 6),
                "savings_usd": round(float(r.savings or 0.0), 6),
            }
        )

    return {"providers": providers_analytics}


def get_savings_breakdown(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Independent savings breakdown across cache, compression, and routing dimensions."""
    # Cache hit savings query
    cache_query = db.query(func.sum(RequestLog.savings_usd)).filter(
        RequestLog.cache_hit.is_(True)
    )
    cache_query = _apply_filters(cache_query, start_date, end_date, tag=tag)
    cache_savings = cache_query.scalar() or 0.0

    # Compression savings query
    compressed_query = db.query(func.sum(RequestLog.savings_usd)).filter(
        RequestLog.compressed.is_(True)
    )
    compressed_query = _apply_filters(compressed_query, start_date, end_date, tag=tag)
    compression_savings = compressed_query.scalar() or 0.0

    # Routing savings query
    routed_query = db.query(func.sum(RequestLog.savings_usd)).filter(
        RequestLog.routed.is_(True)
    )
    routed_query = _apply_filters(routed_query, start_date, end_date, tag=tag)
    routing_savings = routed_query.scalar() or 0.0

    total_query = db.query(func.sum(RequestLog.savings_usd))
    total_query = _apply_filters(total_query, start_date, end_date, tag=tag)
    total_savings = total_query.scalar() or 0.0

    return {
        "cache_savings_usd": round(float(cache_savings), 6),
        "compression_savings_usd": round(float(compression_savings), 6),
        "routing_savings_usd": round(float(routing_savings), 6),
        "total_savings_usd": round(float(total_savings), 6),
    }
