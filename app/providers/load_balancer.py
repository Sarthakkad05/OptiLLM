"""
Load Balancer Strategy Engine.
Selects providers based on configured routing strategy:
  - cost_optimized (default): Selects provider supporting requested model at lowest cost.
  - round_robin: Rotates sequentially across available healthy providers.
  - least_latency: Selects healthy provider with lowest recorded latency.
"""

import logging
from typing import List, Optional

from app.core.circuit_breaker import circuit_breaker
from app.core.config import settings
from app.providers.base import BaseProvider
from app.providers.registry import provider_registry

logger = logging.getLogger("optillm.load_balancer")


class LoadBalancer:
    """Load Balancing Engine managing multi-provider dispatch strategies."""

    def __init__(self):
        self._round_robin_index = 0

    def select_provider(
        self, model: str, strategy: Optional[str] = None
    ) -> Optional[BaseProvider]:
        """
        Selects a healthy, available provider adapter based on routing strategy.
        """
        active_strategy = (strategy or settings.ROUTING_STRATEGY).lower()
        available_names = provider_registry.list_available_providers()
        if not available_names:
            available_names = provider_registry.list_providers()

        # Filter providers by circuit breaker health state
        healthy_providers: List[BaseProvider] = []
        for name in available_names:
            if circuit_breaker.can_execute(name):
                p = provider_registry.get(name)
                if p:
                    healthy_providers.append(p)

        if not healthy_providers:
            logger.warning("No healthy providers available in load balancer.")
            return None

        # Strategy 1: Round Robin
        if active_strategy == "round_robin":
            provider = healthy_providers[
                self._round_robin_index % len(healthy_providers)
            ]
            self._round_robin_index += 1
            logger.info("LoadBalancer [round_robin] selected: %s", provider.name)
            return provider

        # Strategy 2: Least Latency
        if active_strategy == "least_latency":
            # Sort by recorded circuit breaker latency (0ms latency placed last to prefer tested latency)
            sorted_by_latency = sorted(
                healthy_providers,
                key=lambda p: (
                    circuit_breaker.get_circuit(p.name).latency_ms
                    if circuit_breaker.get_circuit(p.name).latency_ms > 0
                    else 999999.0
                ),
            )
            provider = sorted_by_latency[0]
            logger.info("LoadBalancer [least_latency] selected: %s", provider.name)
            return provider

        # Strategy 3: Cost Optimized (default)
        target = provider_registry.get_for_model(model)
        if target and target in healthy_providers:
            logger.info(
                "LoadBalancer [cost_optimized] selected target model provider: %s",
                target.name,
            )
            return target

        # Default fallback to first healthy provider
        provider = healthy_providers[0]
        logger.info("LoadBalancer fallback selected: %s", provider.name)
        return provider


# Global singleton load balancer
load_balancer = LoadBalancer()
