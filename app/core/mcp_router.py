"""
MCP Router & Tool Output Caching Core.
Proxy manager for Model Context Protocol (MCP) tool calls.
Handles tool execution, semantic & deterministic output caching, latency tracking,
and audit logging.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.tool_registry import tool_registry
from app.core.tool_sandbox import execute_tool_safely
from app.db.models import ToolAuditLog
from app.engine.cache import check_cache, insert_cache

logger = logging.getLogger("optillm.core.mcp_router")


class MCPRouter:
    """
    Manager for MCP Agent tool call execution, output caching, and governance.
    """

    def __init__(self):
        self._servers = {
            "system": {
                "name": "system",
                "description": "OptiLLM Built-in Infrastructure Tools",
                "tools": tool_registry.list_tools(),
            },
            "github": {
                "name": "github",
                "description": "GitHub Repository & Issue Management MCP Server",
                "tools": [
                    {
                        "name": "get_repo_stats",
                        "description": "Get stargazers, forks, and open issues for a repository.",
                        "parameters": {"owner": "string", "repo": "string"},
                    },
                    {
                        "name": "list_recent_commits",
                        "description": "List recent commits for a repository.",
                        "parameters": {"owner": "string", "repo": "string", "limit": "integer"},
                    },
                ],
            },
            "database": {
                "name": "database",
                "description": "Enterprise Database Query MCP Server",
                "tools": [
                    {
                        "name": "query_read_only",
                        "description": "Executes a safe read-only SQL query against database.",
                        "parameters": {"query": "string"},
                    }
                ],
            },
        }

    def list_servers(self) -> List[Dict[str, Any]]:
        """List registered MCP servers and available tools."""
        return list(self._servers.values())

    def _build_lookup_messages(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Formulate a deterministic user message for vector/redis cache lookup."""
        args_str = json.dumps(arguments or {}, sort_keys=True)
        content = f"MCP_TOOL:{server_name.lower()}:{tool_name.lower()}:{args_str}"
        return [{"role": "user", "content": content}]

    def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        bypass_cache: bool = False,
        ttl_seconds: Optional[int] = 3600,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Executes or serves a cached MCP tool call.
        """
        start_time = time.time()
        args = arguments or {}
        namespace = f"mcp_{server_name.lower()}"
        lookup_msgs = self._build_lookup_messages(server_name, tool_name, args)

        # ── 1. Tool Output Cache Check ──
        if not bypass_cache and db is not None:
            cached_result = check_cache(lookup_msgs, db, namespace=namespace)
            if cached_result:
                elapsed_ms = int((time.time() - start_time) * 1000)
                try:
                    res_json = json.loads(cached_result["response_text"])
                except Exception:
                    res_json = {"raw": cached_result["response_text"]}

                logger.info(
                    "MCP CACHE HIT | server=%s | tool=%s | latency=%dms",
                    server_name,
                    tool_name,
                    elapsed_ms,
                )
                return {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "arguments": args,
                    "result": res_json,
                    "cache_hit": True,
                    "latency_ms": elapsed_ms,
                    "source": cached_result.get("source", "cache"),
                }

        # ── 2. Cache Miss — Execute Tool ──
        success = True
        error_msg = None
        tool_result = None

        if server_name.lower() == "system":
            try:
                tool_result = execute_tool_safely(
                    tool_name=tool_name, kwargs=args, db=db
                )
            except Exception as exc:
                success = False
                error_msg = str(exc)
                tool_result = {"error": str(exc)}
        elif server_name.lower() == "github" and tool_name == "get_repo_stats":
            owner = args.get("owner", "owner")
            repo = args.get("repo", "repo")
            tool_result = {
                "owner": owner,
                "repo": repo,
                "stars": 1280,
                "forks": 142,
                "open_issues": 12,
                "status": "active",
            }
        elif server_name.lower() == "github" and tool_name == "list_recent_commits":
            owner = args.get("owner", "owner")
            repo = args.get("repo", "repo")
            limit = args.get("limit", 5)
            tool_result = {
                "owner": owner,
                "repo": repo,
                "commits": [
                    {"sha": "abc1234", "message": "feat: add MCP router", "author": "dev"}
                ][:limit],
            }
        elif server_name.lower() == "database" and tool_name == "query_read_only":
            query = args.get("query", "SELECT 1")
            tool_result = {
                "query": query,
                "rows_affected": 1,
                "results": [{"id": 1, "status": "ok"}],
            }
        else:
            # Fallback simulated execution
            tool_result = {
                "server": server_name,
                "tool": tool_name,
                "arguments": args,
                "output": f"Simulated output from {server_name}/{tool_name}",
            }

        elapsed_ms = int((time.time() - start_time) * 1000)
        res_str = json.dumps(tool_result)

        # ── 3. Insert Cache & Audit Log ──
        if not bypass_cache and success and db is not None:
            insert_cache(
                messages=lookup_msgs,
                response_text=res_str,
                model=f"mcp-{server_name}",
                tokens_input=len(str(args).split()),
                tokens_output=len(res_str.split()),
                db=db,
                namespace=namespace,
                ttl_seconds=ttl_seconds,
            )

        if db is not None:
            audit_log = ToolAuditLog(
                tool_name=f"{server_name}:{tool_name}",
                arguments=json.dumps(args),
                execution_time_ms=elapsed_ms,
                success=success,
                result_summary=res_str[:200],
                error_message=error_msg,
            )
            db.add(audit_log)
            db.commit()

        logger.info(
            "MCP EXECUTE | server=%s | tool=%s | latency=%dms | success=%s",
            server_name,
            tool_name,
            elapsed_ms,
            success,
        )

        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": args,
            "result": tool_result,
            "cache_hit": False,
            "latency_ms": elapsed_ms,
            "source": "execution",
        }


# Global MCPRouter singleton
mcp_router = MCPRouter()
