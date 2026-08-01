import pytest


@pytest.mark.asyncio
async def test_list_and_set_budgets_endpoints(async_client):
    # Set custom budget
    payload = {
        "api_key": "team-billing-key",
        "daily_budget_usd": 5.0,
        "monthly_budget_usd": 50.0,
    }
    set_res = await async_client.post("/api/v1/budgets", json=payload)
    assert set_res.status_code == 200
    data = set_res.json()

    assert data["api_key"] == "team-billing-key"
    assert data["daily_budget_usd"] == 5.0

    # Fetch budget by key
    get_res = await async_client.get("/api/v1/budgets/team-billing-key")
    assert get_res.status_code == 200
    assert get_res.json()["monthly_budget_usd"] == 50.0

    # List all budgets
    list_res = await async_client.get("/api/v1/budgets")
    assert list_res.status_code == 200
    assert isinstance(list_res.json(), list)


@pytest.mark.asyncio
async def test_budget_exceeded_rejection(async_client):
    key = "exhausted-budget-key"
    # Set 0 daily budget cap
    await async_client.post(
        "/api/v1/budgets",
        json={"api_key": key, "daily_budget_usd": 0.000001, "monthly_budget_usd": 10.0},
    )

    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Will be blocked by budget"}],
    }

    res = await async_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert res.status_code == 429
    assert "budget" in res.json().get("detail", "").lower()
