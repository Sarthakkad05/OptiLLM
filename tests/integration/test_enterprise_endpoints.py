"""
Integration tests for Phase 14 Enterprise API endpoints.
"""

import pytest


@pytest.mark.asyncio
async def test_tenant_creation_and_audit_log_flow(async_client):
    tenant_payload = {
        "tenant_id": "test_corp_tenant",
        "name": "Test Enterprise Corp",
        "api_key": "sk-ent-test-key-99",
        "role": "developer",
        "sla_target_ms": 300.0,
    }

    # 1. Create tenant (admin role)
    create_res = await async_client.post(
        "/api/v1/enterprise/tenants",
        json=tenant_payload,
        headers={"X-OptiLLM-Tenant-ID": "admin_tenant"},
    )
    assert create_res.status_code == 200
    c_data = create_res.json()
    assert c_data["tenant_id"] == "test_corp_tenant"

    # 2. List tenants
    list_res = await async_client.get(
        "/api/v1/enterprise/tenants",
        headers={"X-OptiLLM-Tenant-ID": "admin_tenant"},
    )
    assert list_res.status_code == 200
    assert any(t["tenant_id"] == "test_corp_tenant" for t in list_res.json())

    # 3. Query audit log and verify hash chain
    audit_res = await async_client.get(
        "/api/v1/enterprise/audit-logs?tenant_id=test_corp_tenant",
        headers={"X-OptiLLM-Tenant-ID": "admin_tenant"},
    )
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["chain_valid"] is True
    assert audit_data["total_entries"] >= 1

    # 4. Check SLA report endpoint
    sla_res = await async_client.get(
        "/api/v1/enterprise/sla?tenant_id=test_corp_tenant",
        headers={"X-OptiLLM-Tenant-ID": "test_corp_tenant"},
    )
    assert sla_res.status_code == 200
    sla_data = sla_res.json()
    assert sla_data["tenant_id"] == "test_corp_tenant"
    assert sla_data["sla_target_ms"] == 300.0
