from app.core.tool_registry import tool_registry
from app.core.tool_sandbox import execute_tool_safely


def test_tool_registry_default_tools():
    tools = tool_registry.list_tools()
    assert len(tools) >= 5

    names = [t["name"] for t in tools]
    assert "get_provider_latency" in names
    assert "get_provider_cost" in names
    assert "get_cache_hit_rate" in names
    assert "get_budget_remaining" in names
    assert "classify_prompt" in names


def test_tool_sandbox_execution(db_session):
    res = execute_tool_safely(
        tool_name="get_provider_cost",
        kwargs={"model": "gpt-4o", "tokens": 500},
        db=db_session,
    )

    assert res["success"] is True
    assert res["tool_name"] == "get_provider_cost"
    assert "cost_usd" in res["result"]


def test_tool_sandbox_custom_registration_and_execution(db_session):
    @tool_registry.register(
        name="test_multiplier_tool", description="Multiplies input by 2"
    )
    def multiply_tool(val: int) -> int:
        return val * 2

    res = execute_tool_safely(
        tool_name="test_multiplier_tool",
        kwargs={"val": 21},
        db=db_session,
    )

    assert res["success"] is True
    assert res["result"] == 42
