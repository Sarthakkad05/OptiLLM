import pytest


@pytest.mark.asyncio
async def test_list_prompts_endpoint(async_client):
    response = await async_client.get("/api/v1/prompts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 3


@pytest.mark.asyncio
async def test_register_and_render_prompts_endpoints(async_client):
    reg_payload = {
        "name": "integration_template",
        "version": "v1",
        "system_template": "System: {sys_val}",
        "user_template": "User: {usr_val}",
    }
    reg_res = await async_client.post("/api/v1/prompts/register", json=reg_payload)
    assert reg_res.status_code == 200

    render_payload = {
        "name": "integration_template",
        "version": "v1",
        "variables": {"sys_val": "Alpha", "usr_val": "Beta"},
    }
    render_res = await async_client.post("/api/v1/prompts/render", json=render_payload)
    assert render_res.status_code == 200
    data = render_res.json()

    assert data["name"] == "integration_template"
    assert len(data["messages"]) == 2
    assert "Alpha" in data["messages"][0]["content"]


@pytest.mark.asyncio
async def test_parse_structured_prompt_endpoint(async_client):
    payload = {"text": '```json\n{"status": "ok"}\n```'}
    response = await async_client.post("/api/v1/prompts/parse", json=payload)
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ok"


@pytest.mark.asyncio
async def test_prompt_versioning_lifecycle(async_client):
    template_name = "versioned_test_template"

    # 1. Register initial v1
    reg_payload = {
        "name": template_name,
        "version": "v1",
        "system_template": "You are a concise {role} assistant.",
        "user_template": "Query: {query}",
        "description": "Initial v1 template",
    }
    reg_res = await async_client.post("/api/v1/prompts/register", json=reg_payload)
    assert reg_res.status_code == 200

    # 2. Create v2 via versions endpoint
    v2_payload = {
        "system_template": "You are an extremely detailed and thorough {role} assistant. Explain everything step by step.",
        "user_template": "User query: {query}. Please provide citations where possible.",
        "description": "Detailed v2 template",
        "commit_message": "Added step-by-step and citations request",
    }
    v2_res = await async_client.post(f"/api/v1/prompts/{template_name}/versions", json=v2_payload)
    assert v2_res.status_code == 200
    v2_data = v2_res.json()
    assert v2_data["version"] == "v2"
    assert v2_data["is_active"] is True

    # 3. List versions history
    hist_res = await async_client.get(f"/api/v1/prompts/{template_name}/versions")
    assert hist_res.status_code == 200
    history = hist_res.json()
    assert len(history) == 2
    versions = [h["version"] for h in history]
    assert "v1" in versions
    assert "v2" in versions

    # 4. Compare v1 and v2
    comp_payload = {
        "version_a": "v1",
        "version_b": "v2",
        "variables": {"role": "math tutor", "query": "What is Euler's identity?"},
    }
    comp_res = await async_client.post(f"/api/v1/prompts/{template_name}/compare", json=comp_payload)
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert comp_data["version_a"]["version"] == "v1"
    assert comp_data["version_b"]["version"] == "v2"
    assert comp_data["token_difference"] > 0
    assert "token_change_percent" in comp_data

    # 5. Rollback to v1
    rb_res = await async_client.post(f"/api/v1/prompts/{template_name}/rollback/v1")
    assert rb_res.status_code == 200
    assert rb_res.json()["active_version"] == "v1"
