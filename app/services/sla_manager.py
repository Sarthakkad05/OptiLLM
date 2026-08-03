"""
SLA Manager — Latency Commitment & Compliance Engine
Tracks per-tenant latency SLAs, identifies breaches, and generates SLA reports.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.db.models import RequestLog, Tenant

logger = logging.getLogger("optillm.services.sla_manager")


class SLAManager:
    """
    Manages tenant SLA commitments and monitors latency compliance.
    """

    def get_sla_report(self, db: Session, tenant_id: str = "default") -> Dict[str, Any]:
        """
        Generate SLA compliance performance report for specified tenant.
        """
        tenant = db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        target_ms = tenant.sla_target_ms if tenant else 500.0

        logs = (
            db.query(RequestLog)
            .filter(RequestLog.tenant_id == tenant_id)
            .all()
        )

        if not logs:
            return {
                "tenant_id": tenant_id,
                "sla_target_ms": target_ms,
                "total_requests": 0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "breach_count": 0,
                "compliance_rate": 100.0,
                "status": "compliant",
            }

        latencies = [l.latency_ms for l in logs if l.latency_ms is not None]
        if not latencies:
            latencies = [0]

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))

        breaches = sum(1 for lat in latencies if lat > target_ms)
        compliance_rate = round((1.0 - (breaches / len(latencies))) * 100.0, 2)

        return {
            "tenant_id": tenant_id,
            "sla_target_ms": target_ms,
            "total_requests": len(latencies),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "breach_count": breaches,
            "compliance_rate": compliance_rate,
            "status": "compliant" if compliance_rate >= 99.0 else "at_risk",
        }


_global_sla_manager: Optional[SLAManager] = None


def get_sla_manager() -> SLAManager:
    global _global_sla_manager
    if _global_sla_manager is None:
        _global_sla_manager = SLAManager()
    return _global_sla_manager
