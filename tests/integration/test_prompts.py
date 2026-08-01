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
