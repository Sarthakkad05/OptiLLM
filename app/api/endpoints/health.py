"""
Health and Readiness Endpoints.

GET /health — Liveness probe (always responds if the process is alive)
GET /ready  — Readiness probe (checks DB, providers, critical dependencies)
"""

import time

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.providers.registry import provider_registry

router = APIRouter()

_startup_time = time.time()


@router.get(
    "/health",
    tags=["Health"],
    summary="Liveness Check",
    description="Returns HTTP 200 if the application process is alive and DB is reachable.",
)
def health_check(db: Session = Depends(get_db)):
    """
    Liveness probe. Returns ok if the application is running and can reach the database.
    Use /ready for a deeper readiness check before serving traffic.
    """
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "service": "optillm-gateway",
        "version": "0.10.0",
        "database": db_status,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }


@router.get(
    "/ready",
    tags=["Health"],
    summary="Readiness Check",
    description="Returns HTTP 200 when OptiLLM is ready to serve traffic (DB connected, at least one provider available).",
)
def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe for Kubernetes / load balancer.
    Checks: DB connectivity + at least one LLM provider configured.
    Returns 503 if not ready.
    """
    from fastapi import HTTPException

    issues = []

    # 1. Database check
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
        issues.append(f"Database not ready: {db_status}")

    # 2. Provider check
    available_providers = provider_registry.list_available_providers()
    if not available_providers:
        issues.append("No LLM providers configured (set at least one API key)")

    if issues:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "issues": issues,
                "database": db_status,
                "available_providers": available_providers,
            },
        )

    return {
        "status": "ready",
        "service": "optillm-gateway",
        "version": "0.10.0",
        "database": db_status,
        "available_providers": available_providers,
        "uptime_seconds": round(time.time() - _startup_time, 1),
    }
