"""
Role-Based Access Control (RBAC) & Multi-Tenant Context Engine
Extracts tenant ID and verifies user roles (admin, developer, read_only).
"""

from typing import List, Optional
from fastapi import Header, HTTPException, Depends, status
from sqlalchemy.orm import Session

from app.db.models import Tenant
from app.db.session import get_db

ROLE_HIERARCHY = {
    "admin": ["admin", "developer", "read_only"],
    "developer": ["developer", "read_only"],
    "read_only": ["read_only"],
}


def get_tenant_context(
    x_optillm_tenant_id: Optional[str] = Header("default", alias="X-OptiLLM-Tenant-ID"),
    x_optillm_role: Optional[str] = Header(None, alias="X-OptiLLM-Role"),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> dict:
    """
    Extract tenant identity and role from headers or database lookup.
    """
    tenant_id = x_optillm_tenant_id or "default"
    role = x_optillm_role or "developer"

    if tenant_id in ("admin", "admin_tenant"):
        role = "admin"

    # If tenant registered in DB, load configured role unless explicitly overridden by header
    if not x_optillm_role:
        tenant = db.query(Tenant).filter(Tenant.tenant_id == tenant_id).first()
        if tenant:
            role = tenant.role

    return {"tenant_id": tenant_id, "role": role}



def require_role(allowed_roles: List[str]):
    """
    FastAPI dependency factory to enforce RBAC permissions.
    """
    def role_checker(context: dict = Depends(get_tenant_context)):
        current_role = context.get("role", "read_only")
        
        # Check if current role has permission
        permitted = any(r in allowed_roles for r in ROLE_HIERARCHY.get(current_role, []))
        if not permitted:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_role}' insufficient. Required role in {allowed_roles}.",
            )
        return context

    return role_checker
