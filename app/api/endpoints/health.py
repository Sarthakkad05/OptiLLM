"""
Health and Readiness Endpoints.

GET /health       — Liveness probe (always 200 if process is alive)
GET /ready        — Readiness probe (K8s-compatible, checks all subsystems)
GET /health/live  — Alias for /health
GET /health/ready — Structured per-subsystem readiness with latency measurements
"""

import time
from typing import Any, Dict, Literal

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.providers.registry import provider_registry

router = APIRouter()

_startup_time = time.time()
_VERSION = "1.0.0"


def _check_database(db: Session) -> Dict[str, Any]:
    """Check DB connectivity and measure round-trip latency."""
    start = time.perf_counter()
    try:
        db.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "ok", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "error", "error": str(exc), "latency_ms": latency_ms}


def _check_redis() -> Dict[str, Any]:
    """Check Redis connectivity and round-trip latency."""
    if not settings.REDIS_URL:
        return {"status": "not_configured"}
    start = time.perf_counter()
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        r.ping()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "ok", "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"status": "error", "error": str(exc), "latency_ms": latency_ms}


def _check_faiss() -> Dict[str, Any]:
    """Check FAISS index availability and size."""
    try:
        from app.engine.faiss_store import get_index, total_vectors
        index = get_index()
        vectors = total_vectors()
        return {
            "status": "ok",
            "vectors": vectors,
            "index_type": type(index).__name__,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _check_providers() -> Dict[str, Any]:
    """Check availability and circuit state of each configured provider."""
    from app.core.circuit_breaker import circuit_breaker

    available = provider_registry.list_available_providers()
    provider_states = {}

    for name in ["openai", "anthropic", "gemini", "groq", "mistral", "azure", "ollama", "bedrock"]:
        circuit_state = circuit_breaker.get_state(name) if hasattr(circuit_breaker, "get_state") else "unknown"
        is_available = name in available
        provider_states[name] = {
            "configured": is_available,
            "circuit": circuit_state,
        }

    return {
        "available_count": len(available),
        "providers": provider_states,
    }


def _check_embedding_model() -> Dict[str, Any]:
    """Check if the embedding model is loaded (required for semantic cache)."""
    try:
        from app.engine.embedding import get_model
        model = get_model()
        return {"status": "ok", "model": str(type(model).__name__)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    tags=["Health"],
    summary="Liveness Check",
    description="Returns HTTP 200 if the process is alive and DB is reachable.",
)
def health_check(db: Session = Depends(get_db)):
    """Simple liveness probe — use /ready for full subsystem check."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "service": "optillm-gateway",
        "version": _VERSION,
        "database": db_status,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@router.get(
    "/ready",
    tags=["Health"],
    summary="Readiness Check",
    description="Returns 200 when ready, 503 when not. Checks DB + at least one provider.",
)
def readiness_check(db: Session = Depends(get_db)):
    """Lightweight readiness probe for Kubernetes/load balancer."""
    from fastapi import HTTPException

    issues = []
    db_check = _check_database(db)
    if db_check["status"] != "ok":
        issues.append(f"Database: {db_check.get('error', 'unknown error')}")

    available_providers = provider_registry.list_available_providers()
    if not available_providers:
        issues.append("No LLM providers configured (set at least one API key)")

    if issues:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "issues": issues,
                "database": db_check,
                "available_providers": available_providers,
            },
        )

    return {
        "status": "ready",
        "service": "optillm-gateway",
        "version": _VERSION,
        "database": db_check,
        "available_providers": available_providers,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@router.get(
    "/health/ready",
    tags=["Health"],
    summary="Detailed Readiness Check",
    description=(
        "Full per-subsystem readiness report with latency measurements. "
        "Checks: Database, Redis, FAISS, Embedding Model, and all provider circuits. "
        "Returns 'ready', 'degraded', or 'down' status."
    ),
)
def detailed_readiness(response: Response, db: Session = Depends(get_db)):
    """
    Structured health check for monitoring dashboards and ops teams.
    Returns per-subsystem status with latencies.

    Status values:
      ready   — all critical subsystems healthy
      degraded — non-critical subsystem(s) unhealthy (still serves traffic)
      down    — critical subsystem(s) unavailable (should not serve traffic)
    """
    checks = {
        "database": _check_database(db),
        "redis": _check_redis(),
        "faiss": _check_faiss(),
        "embedding_model": _check_embedding_model(),
        "providers": _check_providers(),
    }

    # Critical: DB must be healthy
    db_ok = checks["database"]["status"] == "ok"
    # Degraded: Redis/FAISS/embedding issues reduce performance but don't stop serving
    redis_ok = checks["redis"]["status"] in ("ok", "not_configured")
    faiss_ok = checks["faiss"]["status"] == "ok"
    providers_ok = checks["providers"]["available_count"] > 0

    if not db_ok or not providers_ok:
        overall = "down"
        response.status_code = 503
    elif not redis_ok or not faiss_ok:
        overall = "degraded"
        # Still 200 — degraded but serving
    else:
        overall = "ready"

    return {
        "status": overall,
        "service": "optillm-gateway",
        "version": _VERSION,
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "checks": checks,
    }


@router.get("/health/live", tags=["Health"], summary="Liveness alias")
def liveness(db: Session = Depends(get_db)):
    """Alias for /health — Kubernetes liveness probe."""
    return health_check(db)
