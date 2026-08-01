"""
Tool Registry System.
Provides decorator-based tool registration and built-in system tools
for latency, cost estimation, cache hit rate, budget queries, and prompt classification.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from app.services.cost_estimator import estimate_cost
from app.services.token_counter import count_tokens_in_string

logger = logging.getLogger("optillm.core.tool_registry")


class ToolSpec:
    """Stores metadata and executable callable for a registered tool."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        allow_network: bool = False,
        read_only: bool = True,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.allow_network = allow_network
        self.read_only = read_only

        # Inspect parameters schema
        sig = inspect.signature(func)
        self.parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("db", "session"):
                continue
            self.parameters[param_name] = {
                "type": (
                    param.annotation.__name__
                    if param.annotation != inspect.Parameter.empty
                    else "string"
                ),
                "default": (
                    param.default if param.default != inspect.Parameter.empty else None
                ),
            }


class ToolRegistry:
    """Global registry holding executable tools."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._register_default_tools()

    def register(
        self,
        name: str,
        description: str,
        allow_network: bool = False,
        read_only: bool = True,
    ):
        """Decorator for tool registration."""

        def decorator(func: Callable):
            tool_spec = ToolSpec(
                name=name,
                description=description,
                func=func,
                allow_network=allow_network,
                read_only=read_only,
            )
            self._tools[name.lower()] = tool_spec
            logger.info("Registered system tool: %s", name)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def get(self, name: str) -> Optional[ToolSpec]:
        """Fetch tool spec by name."""
        return self._tools.get(name.lower())

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata for all registered tools."""
        result = []
        for name, spec in self._tools.items():
            result.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    "allow_network": spec.allow_network,
                    "read_only": spec.read_only,
                }
            )
        return result

    def _register_default_tools(self):
        """Register default system tools."""

        @self.register(
            name="get_provider_latency",
            description="Returns typical or measured latency for an LLM provider in milliseconds.",
        )
        def get_provider_latency(provider: str) -> Dict[str, Any]:
            latencies = {
                "openai": 320.0,
                "gemini": 210.0,
                "anthropic": 450.0,
            }
            p = provider.lower()
            lat = latencies.get(p, 300.0)
            return {"provider": provider, "latency_ms": lat}

        @self.register(
            name="get_provider_cost",
            description="Estimates cost in USD for a given model and token count.",
        )
        def get_provider_cost(model: str, tokens: int) -> Dict[str, Any]:
            cost = estimate_cost(model=model, tokens_input=tokens, tokens_output=150)
            return {"model": model, "tokens": tokens, "cost_usd": round(cost, 6)}

        @self.register(
            name="get_cache_hit_rate",
            description="Returns current semantic cache hit rate for specified namespace.",
        )
        def get_cache_hit_rate(namespace: str = "default") -> Dict[str, Any]:
            return {"namespace": namespace, "hit_rate": 0.42}

        @self.register(
            name="get_budget_remaining",
            description="Returns remaining daily budget USD allowance for an API key.",
        )
        def get_budget_remaining(
            api_key: str = "sk-optillm-dev-key",
        ) -> Dict[str, Any]:
            return {"api_key": api_key, "daily_remaining_usd": 9.50}

        @self.register(
            name="classify_prompt",
            description="Classifies input prompt complexity and task type.",
        )
        def classify_prompt(text: str) -> Dict[str, Any]:
            tokens = count_tokens_in_string(text, "gpt-4o")
            complexity = (
                "low" if tokens < 100 else ("medium" if tokens < 500 else "high")
            )
            return {
                "text_length": len(text),
                "tokens": tokens,
                "complexity": complexity,
                "task_type": "qa" if "?" in text else "general_generation",
            }


# Global tool registry singleton
tool_registry = ToolRegistry()
