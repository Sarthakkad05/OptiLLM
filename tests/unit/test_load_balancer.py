from app.providers.load_balancer import LoadBalancer
from app.providers.registry import provider_registry


def test_load_balancer_cost_optimized_strategy():
    lb = LoadBalancer()
    provider = lb.select_provider(model="gpt-4o", strategy="cost_optimized")
    assert provider is not None
    assert provider.name == "openai"


def test_load_balancer_round_robin_strategy():
    lb = LoadBalancer()
    p1 = lb.select_provider(model="gpt-4o", strategy="round_robin")
    p2 = lb.select_provider(model="gpt-4o", strategy="round_robin")

    assert p1 is not None
    assert p2 is not None
    available = provider_registry.list_available_providers()
    if len(available) > 1:
        assert p1.name != p2.name


def test_load_balancer_least_latency_strategy():
    lb = LoadBalancer()
    provider = lb.select_provider(model="gpt-4o", strategy="least_latency")
    assert provider is not None
