import pytest

from app.engine.parallel_eval import run_parallel_model_comparison


@pytest.mark.asyncio
async def test_parallel_model_comparison():
    messages = [{"role": "user", "content": "Benchmark test prompt"}]
    res = await run_parallel_model_comparison(
        messages=messages,
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
    )

    assert "winner_model" in res
    assert "winner_content" in res
    assert "model_a_result" in res
    assert "model_b_result" in res
    assert res["winner_model"] in ["gpt-4o", "gpt-4o-mini"]
