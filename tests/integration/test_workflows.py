import pytest


@pytest.mark.asyncio
async def test_workflow_graph_inspection_endpoint(async_client):
    response = await async_client.get("/api/v1/workflows/graph")
    assert response.status_code == 200
    data = response.json()

    assert "graph_name" in data
    assert "nodes" in data
    assert "check_cache" in data["nodes"]
    assert "conditional_edges" in data


@pytest.mark.asyncio
async def test_workflow_run_endpoint(async_client):
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Execute workflow endpoint test"}],
    }
    response = await async_client.post("/api/v1/workflows/run", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "content" in data
    assert "quality_score" in data


@pytest.mark.asyncio
async def test_workflow_compare_endpoint(async_client):
    payload = {
        "model_a": "gpt-4o",
        "model_b": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Compare models benchmark"}],
    }
    response = await async_client.post("/api/v1/workflows/compare", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "winner_model" in data
    assert "comparison_metrics" in data
