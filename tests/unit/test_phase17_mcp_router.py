"""
Unit & Integration Tests for Phase 17 — MCP Agent Tool Router & Output Caching
Tests server discovery, tool execution, tool output caching, and cache clearing endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.mcp_router import mcp_router
from app.main import app

client = TestClient(app)


def test_list_mcp_servers_endpoint():
    """Verify GET /api/v1/mcp/servers returns registered servers and tools."""
    res = client.get("/api/v1/mcp/servers")
    assert res.status_code == 200
    servers = res.json()
    assert len(servers) >= 3
    server_names = [s["name"] for s in servers]
    assert "system" in server_names
    assert "github" in server_names
    assert "database" in server_names


def test_proxy_mcp_tool_call_and_cache_hit():
    """Verify POST /api/v1/mcp/proxy executes tool on first call and hits cache on second call."""
    payload = {
        "server_name": "github",
        "tool_name": "get_repo_stats",
        "arguments": {"owner": "Sarthakkad05", "repo": "OptiLLM"},
        "optillm": {"bypass_cache": False, "ttl_seconds": 3600},
    }

    # 1. First invocation (Execution)
    res1 = client.post("/api/v1/mcp/proxy", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["server_name"] == "github"
    assert data1["tool_name"] == "get_repo_stats"
    assert data1["result"]["stars"] == 1280
    assert data1["cache_hit"] is False

    # 2. Second invocation (Cache Hit)
    res2 = client.post("/api/v1/mcp/proxy", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["result"]["stars"] == 1280


def test_clear_mcp_cache_endpoint():
    """Verify DELETE /api/v1/mcp/cache/clear wipes MCP tool output cache."""
    res = client.delete("/api/v1/mcp/cache/clear?server_name=github")
    assert res.status_code == 200
    assert "entries_removed" in res.json()
