"""
Cost Estimation Service
Calculates USD cost based on provider pricing tables.
All prices are per 1,000,000 tokens (as published by providers).

Sources (approximate, update as needed):
  - OpenAI:  https://openai.com/pricing
  - Google:  https://ai.google.dev/pricing
"""

from typing import Tuple
import logging

logger = logging.getLogger("optillm.cost_estimator")

# Pricing: {model: (input_price_per_1M, output_price_per_1M)} in USD
PRICING_TABLE: dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o":               (5.00,   15.00),
    "gpt-4o-mini":          (0.15,    0.60),
    "gpt-4":                (30.00,  60.00),
    "gpt-4-turbo":          (10.00,  30.00),
    "gpt-3.5-turbo":        (0.50,   1.50),

    # Google Gemini
    "gemini-1.5-pro":       (3.50,   10.50),
    "gemini-1.5-flash":     (0.35,    1.05),
    "gemini-2.0-flash":     (0.10,    0.40),
    "gemini-1.0-pro":       (0.50,   1.50),
}

# Default fallback when model is unknown
_FALLBACK_PRICING: Tuple[float, float] = (5.00, 15.00)


def get_model_pricing(model: str) -> Tuple[float, float]:
    """Returns (input_price_per_1M, output_price_per_1M) for a model."""
    # Try exact match first, then prefix match
    if model in PRICING_TABLE:
        return PRICING_TABLE[model]
    for key in PRICING_TABLE:
        if model.startswith(key) or key.startswith(model):
            return PRICING_TABLE[key]
    logger.warning("Unknown model '%s' — using fallback pricing (gpt-4o rates).", model)
    return _FALLBACK_PRICING


def estimate_cost(model: str, tokens_input: int, tokens_output: int) -> float:
    """Calculate USD cost for a given model and token counts."""
    input_price, output_price = get_model_pricing(model)
    cost = (tokens_input / 1_000_000) * input_price + (tokens_output / 1_000_000) * output_price
    return round(cost, 8)


def estimate_cache_savings(model: str, tokens_input: int, tokens_output: int) -> float:
    """
    Savings when a cache hit occurs.
    We saved 100% of the cost since no LLM call was made.
    """
    return estimate_cost(model, tokens_input, tokens_output)


def estimate_compression_savings(
    model: str, original_tokens: int, compressed_tokens: int
) -> float:
    """Savings from reducing input tokens via context compression."""
    tokens_saved = max(0, original_tokens - compressed_tokens)
    input_price, _ = get_model_pricing(model)
    return round((tokens_saved / 1_000_000) * input_price, 8)


def estimate_routing_savings(
    original_model: str, routed_model: str, tokens_input: int, tokens_output: int
) -> float:
    """
    Savings from routing to a cheaper model.
    Savings = (cost at original model) - (cost at routed model)
    """
    original_cost = estimate_cost(original_model, tokens_input, tokens_output)
    routed_cost = estimate_cost(routed_model, tokens_input, tokens_output)
    return round(max(0.0, original_cost - routed_cost), 8)
