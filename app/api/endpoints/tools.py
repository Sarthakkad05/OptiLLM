"""
GET /api/v1/tools
POST /api/v1/tools/execute
GET /api/v1/tools/audit
Tool discovery, sandboxed execution, and audit logging API endpoints.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.tool_registry import tool_registry
from app.core.tool_sandbox import execute_tool_safely
from app.db.models import ToolAuditLog
from app.db.session import get_db

router = APIRouter()


class ExecuteToolRequest(BaseModel):
    tool_name: str
    kwargs: Optional[Dict[str, Any]] = None


@router.get("", tags=["Tools"])
def list_tools() -> List[Dict[str, Any]]:
    """List all registered system tools, parameter schemas, and security attributes."""
    return tool_registry.list_tools()


@router.post("/execute", tags=["Tools"])
def execute_tool_endpoint(
    request: ExecuteToolRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Executes a registered tool within a sandboxed executor with 2-second timeout protection.
    """
    try:
        return execute_tool_safely(
            tool_name=request.tool_name,
            kwargs=request.kwargs or {},
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/audit", tags=["Tools"])
def list_tool_audit_logs(
    limit: int = 50, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Fetch recent tool execution audit logs."""
    logs = (
        db.query(ToolAuditLog)
        .order_by(ToolAuditLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    result = []
    for log_item in logs:
        result.append(
            {
                "id": log_item.id,
                "timestamp": (
                    log_item.timestamp.isoformat() if log_item.timestamp else None
                ),
                "tool_name": log_item.tool_name,
                "arguments": log_item.arguments,
                "execution_time_ms": log_item.execution_time_ms,
                "success": log_item.success,
                "result_summary": log_item.result_summary,
                "error_message": log_item.error_message,
            }
        )

    return result
