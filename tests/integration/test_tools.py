import pytest


@pytest.mark.asyncio
async def test_list_tools_endpoint(async_client):
    response = await async_client.get("/api/v1/tools")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 5


@pytest.mark.asyncio
async def test_execute_tool_endpoint(async_client):
    payload = {
        "tool_name": "classify_prompt",
        "kwargs": {"text": "What is quantum mechanics?"},
    }
    response = await async_client.post("/api/v1/tools/execute", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["result"]["complexity"] == "low"

    # Check audit log
    audit_res = await async_client.get("/api/v1/tools/audit")
    assert audit_res.status_code == 200
    audit_logs = audit_res.json()
    assert isinstance(audit_logs, list)
    assert len(audit_logs) >= 1
