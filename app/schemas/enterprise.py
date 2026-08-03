"""
Pydantic Schemas for Phase 14 Enterprise Features
Defines schemas for tenant management, RBAC, audit log verification, and SLA reporting.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CreateTenantRequest(BaseModel):
    tenant_id: str = Field(..., description="Unique tenant identifier (slug)")
    name: str = Field(..., description="Organization or team name")
    api_key: str = Field(..., description="Tenant API Key")
    role: str = Field("developer", description="Role: admin | developer | read_only")
    sla_target_ms: float = Field(500.0, ge=10.0, le=10000.0, description="P95 Latency SLA target in ms")


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    role: str
    sla_target_ms: float
    status: str = "active"


class AuditLogQueryResponse(BaseModel):
    tenant_id: str
    total_entries: int
    chain_valid: bool
    entries: List[Dict[str, Any]]


class PluginToggleRequest(BaseModel):
    plugin_name: str
    enabled: bool


class SLAReportResponse(BaseModel):
    tenant_id: str
    sla_target_ms: float
    total_requests: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    breach_count: int
    compliance_rate: float
    status: str
