"""
Enterprise API Endpoints — Multi-tenancy, RBAC, Audit Logs, Plugins, and SLA Reports
Exposes REST endpoints for managing tenants, checking audit log hash chains, controlling plugins, and monitoring SLAs.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.rbac import require_role
from app.db.models import AuditLogEntry, Tenant
from app.db.session import get_db
from app.plugins.registry import get_plugin_registry
from app.schemas.enterprise import (
    AuditLogQueryResponse,
    CreateTenantRequest,
    PluginToggleRequest,
    SLAReportResponse,
    TenantResponse,
)
from app.security.audit import get_audit_logger
from app.services.sla_manager import get_sla_manager

router = APIRouter(prefix="/enterprise", tags=["Enterprise"])


@router.post(
    "/tenants",
    response_model=TenantResponse,
    summary="Create enterprise tenant",
    dependencies=[Depends(require_role(["admin"]))],
)
def create_tenant(
    payload: CreateTenantRequest, db: Session = Depends(get_db)
) -> TenantResponse:
    """
    Registers a new tenant with assigned RBAC role and SLA latency target.
    Requires 'admin' role.
    """
    existing = db.query(Tenant).filter(Tenant.tenant_id == payload.tenant_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant '{payload.tenant_id}' already exists.",
        )

    tenant = Tenant(
        tenant_id=payload.tenant_id,
        name=payload.name,
        api_key=payload.api_key,
        role=payload.role,
        sla_target_ms=payload.sla_target_ms,
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    # Record tamper-evident audit log
    audit_logger = get_audit_logger()
    audit_logger.record_action(
        db,
        tenant_id=payload.tenant_id,
        actor="admin",
        action="create_tenant",
        resource=f"tenant:{payload.tenant_id}",
        payload={"role": payload.role, "sla_target_ms": payload.sla_target_ms},
    )

    return TenantResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        role=tenant.role,
        sla_target_ms=tenant.sla_target_ms,
        status="active",
    )


@router.get(
    "/tenants",
    response_model=List[TenantResponse],
    summary="List enterprise tenants",
    dependencies=[Depends(require_role(["admin"]))],
)
def list_tenants(db: Session = Depends(get_db)) -> List[TenantResponse]:
    """List all registered tenants. Requires 'admin' role."""
    tenants = db.query(Tenant).all()
    return [
        TenantResponse(
            tenant_id=t.tenant_id,
            name=t.name,
            role=t.role,
            sla_target_ms=t.sla_target_ms,
            status="active",
        )
        for t in tenants
    ]


@router.get(
    "/audit-logs",
    response_model=AuditLogQueryResponse,
    summary="Query audit log and verify hash chain",
    dependencies=[Depends(require_role(["admin"]))],
)
def get_audit_logs(
    tenant_id: str = Query("default", description="Target tenant ID"),
    db: Session = Depends(get_db),
) -> AuditLogQueryResponse:
    """
    Fetch immutable audit log entries and verify cryptographic SHA256 chain integrity.
    Requires 'admin' role.
    """
    audit_logger = get_audit_logger()
    integrity = audit_logger.verify_chain_integrity(db, tenant_id=tenant_id)

    entries_raw = (
        db.query(AuditLogEntry)
        .filter(AuditLogEntry.tenant_id == tenant_id)
        .order_by(AuditLogEntry.id.desc())
        .all()
    )

    entries = [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat() if e.timestamp else "",
            "actor": e.actor,
            "action": e.action,
            "resource": e.resource,
            "payload_hash": e.payload_hash,
            "prev_hash": e.prev_hash,
            "status": e.status,
        }
        for e in entries_raw
    ]

    return AuditLogQueryResponse(
        tenant_id=tenant_id,
        total_entries=len(entries),
        chain_valid=integrity["valid"],
        entries=entries,
    )


@router.get(
    "/plugins",
    summary="List registered gateway plugins",
    dependencies=[Depends(require_role(["admin", "developer"]))],
)
def list_plugins() -> Dict[str, Any]:
    """List enterprise plugins and status."""
    registry = get_plugin_registry()
    return {"plugins": registry.list_plugins()}


@router.post(
    "/plugins/toggle",
    summary="Toggle plugin status",
    dependencies=[Depends(require_role(["admin"]))],
)
def toggle_plugin(payload: PluginToggleRequest) -> Dict[str, Any]:
    """Enable or disable plugin at runtime. Requires 'admin' role."""
    registry = get_plugin_registry()
    success = registry.toggle(payload.plugin_name, payload.enabled)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_44_NOT_FOUND if hasattr(status, "HTTP_44_NOT_FOUND") else 404,
            detail=f"Plugin '{payload.plugin_name}' not found.",
        )
    return {
        "plugin_name": payload.plugin_name,
        "enabled": payload.enabled,
        "status": "updated",
    }


@router.get(
    "/sla",
    response_model=SLAReportResponse,
    summary="Get SLA latency compliance report",
    dependencies=[Depends(require_role(["admin", "developer", "read_only"]))],
)
def get_sla_report(
    tenant_id: str = Query("default", description="Tenant ID"),
    db: Session = Depends(get_db),
) -> SLAReportResponse:
    """Returns SLA compliance percentage, latency percentiles, and breach counts."""
    sla_manager = get_sla_manager()
    report = sla_manager.get_sla_report(db, tenant_id=tenant_id)
    return SLAReportResponse(**report)
