from app.services.cost_estimator import (
    estimate_cache_savings,
    estimate_compression_savings,
    estimate_cost,
    estimate_routing_savings,
)


def test_estimate_cost():
    cost = estimate_cost("gpt-3.5-turbo", tokens_input=1000, tokens_output=500)
    assert cost > 0.0
    assert isinstance(cost, float)


def test_estimate_cache_savings():
    savings = estimate_cache_savings("gpt-4o", tokens_input=1000, tokens_output=500)
    assert savings > 0.0


def test_estimate_compression_savings():
    savings = estimate_compression_savings(
        model="gpt-4o", original_tokens=2000, compressed_tokens=1000
    )
    assert savings > 0.0


def test_estimate_routing_savings():
    savings = estimate_routing_savings(
        original_model="gpt-4o",
        routed_model="gpt-3.5-turbo",
        tokens_input=1000,
        tokens_output=500,
    )
    assert savings > 0.0
