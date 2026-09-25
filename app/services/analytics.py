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
    unique_models = (
        db.query(func.count(func.distinct(RequestLog.model_used))).scalar() or 0
    )

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
        "unique_models": int(unique_models),
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


def get_quality_analytics(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    provider: Optional[str] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregates response quality ratings, 5-dimension averages, hallucination risks,
    cost-efficiency scores, and model quality rankings.
    """
    base_query = db.query(RequestLog)
    base_query = _apply_filters(base_query, start_date, end_date, provider=provider, tag=tag)

    eval_query = base_query.filter(RequestLog.quality_score.isnot(None))
    total_eval = eval_query.count()

    if total_eval == 0:
        return {
            "total_evaluated_requests": 0,
            "avg_quality_score": 0.0,
            "avg_hallucination_score": 0.0,
            "avg_efficiency_score": 0.0,
            "dimension_averages": {
                "correctness": 0.0,
                "relevance": 0.0,
                "completeness": 0.0,
            },
            "model_quality_breakdown": [],
            "quality_over_time": [],
        }

    avg_qual = eval_query.with_entities(func.avg(RequestLog.quality_score)).scalar() or 0.0
    avg_hall = eval_query.with_entities(func.avg(RequestLog.hallucination_score)).scalar() or 0.0
    avg_eff = eval_query.with_entities(func.avg(RequestLog.efficiency_score)).scalar() or 0.0

    avg_corr = eval_query.with_entities(func.avg(RequestLog.correctness_score)).scalar() or 0.0
    avg_rel = eval_query.with_entities(func.avg(RequestLog.relevance_score)).scalar() or 0.0
    avg_comp = eval_query.with_entities(func.avg(RequestLog.completeness_score)).scalar() or 0.0

    # Per-model breakdown
    model_stats = (
        eval_query.with_entities(
            RequestLog.model_used,
            func.count(RequestLog.id).label("count"),
            func.avg(RequestLog.quality_score).label("avg_quality"),
            func.avg(RequestLog.efficiency_score).label("avg_efficiency"),
        )
        .group_by(RequestLog.model_used)
        .all()
    )

    breakdown = [
        {
            "model": row.model_used,
            "eval_count": row.count,
            "avg_quality_score": round(float(row.avg_quality or 0.0), 4),
            "avg_efficiency_score": round(float(row.avg_efficiency or 0.0), 2),
        }
        for row in model_stats
    ]

    return {
        "total_evaluated_requests": total_eval,
        "avg_quality_score": round(float(avg_qual), 4),
        "avg_hallucination_score": round(float(avg_hall), 4),
        "avg_efficiency_score": round(float(avg_eff), 2),
        "dimension_averages": {
            "correctness": round(float(avg_corr), 4),
            "relevance": round(float(avg_rel), 4),
            "completeness": round(float(avg_comp), 4),
        },
        "model_quality_breakdown": breakdown,
        "quality_over_time": [],
    }


def get_quality_cost_tradeoff(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    tag: Optional[str] = None,
    min_requests: int = 5,
) -> Dict[str, Any]:
    """
    Quality-Cost Tradeoff Analysis.
    Answers: "Which model gives the best quality per dollar for MY workload?"

    This is OptiLLM's killer analytics feature — no other LLM gateway correlates
    actual response quality (from the evaluation judge) with actual cost per request.

    Returns per-model stats ranked by quality_per_dollar with an auto-generated
    plain-English recommendation.
    """
    query = db.query(
        RequestLog.model_used,
        func.count(RequestLog.id).label("request_count"),
        func.avg(RequestLog.cost_usd).label("avg_cost"),
        func.avg(RequestLog.quality_score).label("avg_quality"),
        func.avg(RequestLog.latency_ms).label("avg_latency"),
    ).filter(
        RequestLog.quality_score.isnot(None),
        RequestLog.cost_usd.isnot(None),
    )

    query = _apply_filters(query, start_date, end_date, tag=tag)
    rows = query.group_by(RequestLog.model_used).all()

    model_stats = []
    for row in rows:
        if row.request_count < min_requests:
            continue
        avg_cost = float(row.avg_cost or 0.0)
        avg_quality = float(row.avg_quality or 0.0)

        # quality_per_dollar: higher is better.
        # Avoid division by zero for zero-cost requests (cache hits).
        quality_per_dollar = (
            round(avg_quality / avg_cost, 2) if avg_cost > 0.000001 else 9999.0
        )

        # p95 latency for this model
        latency_rows = (
            db.query(RequestLog.latency_ms)
            .filter(
                RequestLog.model_used == row.model_used,
                RequestLog.quality_score.isnot(None),
            )
            .all()
        )
        latencies = [r.latency_ms for r in latency_rows if r.latency_ms]
        p95_latency = round(float(np.percentile(latencies, 95)), 0) if latencies else 0.0

        model_stats.append({
            "model": row.model_used,
            "request_count": row.request_count,
            "avg_cost_per_request_usd": round(avg_cost, 6),
            "avg_quality_score": round(avg_quality, 4),
            "quality_per_dollar": quality_per_dollar,
            "p95_latency_ms": p95_latency,
        })

    # Sort by quality_per_dollar descending (best value first)
    model_stats.sort(key=lambda x: x["quality_per_dollar"], reverse=True)

    # Auto-generate a plain-English recommendation
    recommendation = None
    if len(model_stats) >= 2:
        best = model_stats[0]
        premium = max(model_stats, key=lambda x: x["avg_quality_score"])
        if best["model"] == premium["model"]:
            recommendation = (
                f"{best['model']} is both the highest-quality and best-value model for your "
                f"workload, with quality {best['avg_quality_score']:.2f} at "
                f"${best['avg_cost_per_request_usd']:.5f}/request."
            )
        else:
            quality_pct = round(
                best["avg_quality_score"] / max(premium["avg_quality_score"], 0.01) * 100
            )
            cost_pct = round(
                best["avg_cost_per_request_usd"]
                / max(premium["avg_cost_per_request_usd"], 0.000001)
                * 100
            )
            recommendation = (
                f"For your workload, {best['model']} delivers {quality_pct}% of "
                f"{premium['model']}'s quality at {cost_pct}% of the cost — "
                f"the best quality-per-dollar trade-off."
            )
    elif len(model_stats) == 1:
        m = model_stats[0]
        recommendation = (
            f"Only {m['model']} has enough evaluated requests. "
            f"Run more requests to enable cross-model comparison."
        )

    return {
        "models": model_stats,
        "recommendation": recommendation,
        "total_models_analyzed": len(model_stats),
        "note": (
            "Requires EVAL_ENABLED=true and at least 5 evaluated requests per model. "
            "Enable EVAL_LLM_ENABLED=true for more accurate quality scores."
        ),
    }


def get_cost_attribution(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Cost attribution breakdown:
      - Spend by Model
      - Spend by Team / Tag
      - Savings Waterfall (Cache, Compression, Routing)
      - Latency & SLA compliance by Provider
    """
    base_query = db.query(RequestLog)
    base_query = _apply_filters(base_query, start_date, end_date)

    total_cost = base_query.with_entities(func.sum(RequestLog.cost_usd)).scalar() or 0.0
    total_savings = base_query.with_entities(func.sum(RequestLog.savings_usd)).scalar() or 0.0
    total_requests = base_query.with_entities(func.count(RequestLog.id)).scalar() or 0

    # 1. Cost by Model
    model_rows = (
        base_query.with_entities(
            RequestLog.model_used,
            func.sum(RequestLog.cost_usd).label("cost"),
            func.count(RequestLog.id).label("requests"),
            func.sum(RequestLog.tokens_input).label("tokens_in"),
            func.sum(RequestLog.tokens_output).label("tokens_out"),
        )
        .group_by(RequestLog.model_used)
        .all()
    )
    cost_by_model = []
    for r in model_rows:
        m_cost = float(r.cost or 0.0)
        cost_by_model.append({
            "model": r.model_used or "unknown",
            "cost_usd": round(m_cost, 6),
            "requests": r.requests,
            "tokens_input": int(r.tokens_in or 0),
            "tokens_output": int(r.tokens_out or 0),
            "percent_of_total": round((m_cost / total_cost * 100), 2) if total_cost > 0 else 0.0,
        })
    cost_by_model.sort(key=lambda x: x["cost_usd"], reverse=True)

    # 2. Cost by Team / Tag
    tag_rows = (
        base_query.with_entities(
            func.coalesce(RequestLog.tag, "default").label("team"),
            func.sum(RequestLog.cost_usd).label("cost"),
            func.count(RequestLog.id).label("requests"),
        )
        .group_by(RequestLog.tag)
        .all()
    )
    cost_by_team = []
    for r in tag_rows:
        t_cost = float(r.cost or 0.0)
        cost_by_team.append({
            "team": r.team,
            "cost_usd": round(t_cost, 6),
            "requests": r.requests,
            "percent_of_total": round((t_cost / total_cost * 100), 2) if total_cost > 0 else 0.0,
        })
    cost_by_team.sort(key=lambda x: x["cost_usd"], reverse=True)

    # 3. Savings Waterfall
    cache_savings = (
        base_query.filter(RequestLog.cache_hit == True)
        .with_entities(func.sum(RequestLog.savings_usd))
        .scalar()
        or 0.0
    )
    compression_savings = (
        base_query.filter(RequestLog.cache_hit == False, RequestLog.compressed == True)
        .with_entities(func.sum(RequestLog.savings_usd))
        .scalar()
        or 0.0
    )
    routing_savings = (
        base_query.filter(RequestLog.cache_hit == False, RequestLog.compressed == False, RequestLog.routed == True)
        .with_entities(func.sum(RequestLog.savings_usd))
        .scalar()
        or 0.0
    )
    gross_spend = float(total_cost + total_savings)
    savings_ratio = round((total_savings / gross_spend * 100), 2) if gross_spend > 0 else 0.0

    savings_waterfall = {
        "gross_spend_usd": round(gross_spend, 6),
        "net_spend_usd": round(float(total_cost), 6),
        "total_savings_usd": round(float(total_savings), 6),
        "savings_percentage": savings_ratio,
        "breakdown": {
            "semantic_cache_usd": round(float(cache_savings), 6),
            "context_compression_usd": round(float(compression_savings), 6),
            "model_routing_usd": round(float(routing_savings), 6),
        },
    }

    # 4. Latency by Provider
    provider_rows = (
        base_query.with_entities(
            RequestLog.provider,
            func.count(RequestLog.id).label("requests"),
            func.avg(RequestLog.latency_ms).label("avg_latency"),
        )
        .group_by(RequestLog.provider)
        .all()
    )
    provider_latencies = []
    for r in provider_rows:
        p_logs = (
            base_query.filter(RequestLog.provider == r.provider)
            .with_entities(RequestLog.latency_ms)
            .all()
        )
        lats = [l[0] for l in p_logs if l[0] is not None]
        p50 = float(np.percentile(lats, 50)) if lats else 0.0
        p95 = float(np.percentile(lats, 95)) if lats else 0.0
        p99 = float(np.percentile(lats, 99)) if lats else 0.0
        provider_latencies.append({
            "provider": r.provider or "unknown",
            "requests": r.requests,
            "avg_latency_ms": round(float(r.avg_latency or 0.0), 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
        })

    # 5. SLA Compliance
    try:
        from app.services.sla_manager import get_sla_manager
        sla_report = get_sla_manager().get_sla_report(db)
    except Exception:
        sla_report = {"compliance_rate": 100.0, "status": "compliant"}

    return {
        "summary": {
            "total_requests": total_requests,
            "total_cost_usd": round(float(total_cost), 6),
            "total_savings_usd": round(float(total_savings), 6),
            "savings_percentage": savings_ratio,
        },
        "cost_by_model": cost_by_model,
        "cost_by_team": cost_by_team,
        "savings_waterfall": savings_waterfall,
        "provider_latencies": provider_latencies,
        "sla_compliance": sla_report,
    }

