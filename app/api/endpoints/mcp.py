"""
POST /api/v1/mcp/proxy
GET  /api/v1/mcp/servers
DELETE /api/v1/mcp/cache/clear

MCP Agent Tool Proxy, Caching, and Server Discovery endpoints.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.mcp_router import mcp_router
from app.db.session import get_db
from app.engine.cache import clear_cache

router = APIRouter()


class MCPOptiLLMOptions(BaseModel):
    bypass_cache: bool = False
    ttl_seconds: Optional[int] = 3600


class MCPProxyRequest(BaseModel):
    server_name: str = Field(..., description="Target MCP server name (e.g. system, github, database)")
    tool_name: str = Field(..., description="Tool function name (e.g. get_repo_stats, query_read_only)")
    arguments: Optional[Dict[str, Any]] = Field(default_factory=dict)
    optillm: Optional[MCPOptiLLMOptions] = MCPOptiLLMOptions()


class MCPProxyResponse(BaseModel):
    server_name: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    cache_hit: bool
    latency_ms: int
    source: str


@router.get("/servers", tags=["MCP"])
def list_mcp_servers() -> List[Dict[str, Any]]:
    """List registered MCP servers and available tools."""
    return mcp_router.list_servers()


@router.post(
    "/proxy",
    response_model=MCPProxyResponse,
    tags=["MCP"],
    summary="Proxy Agent MCP Tool Call",
    description="Proxies an MCP tool call with output semantic caching, execution sandboxing, and audit logging.",
)
def proxy_mcp_tool_call(
    request: MCPProxyRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Execute or serve cached MCP tool call."""
    opt_cfg = request.optillm or MCPOptiLLMOptions()
    return mcp_router.execute_tool(
        server_name=request.server_name,
        tool_name=request.tool_name,
        arguments=request.arguments,
        bypass_cache=opt_cfg.bypass_cache,
        ttl_seconds=opt_cfg.ttl_seconds,
        db=db,
    )


@router.delete("/cache/clear", tags=["MCP"])
def clear_mcp_cache(
    server_name: Optional[str] = None, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Wipe MCP tool output cache."""
    namespace = f"mcp_{server_name.lower()}" if server_name else None
    count = clear_cache(db, namespace=namespace)
    return {"message": "MCP cache cleared", "entries_removed": count}
