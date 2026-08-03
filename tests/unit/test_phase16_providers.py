"""
Unit tests for Phase 16 — Provider Expansion (Ollama & Anthropic)
Verifies Ollama and Anthropic provider adapters, model matching,
cost estimation ($0.00 for local Ollama), and registry lookup.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.providers.anthropic_client import anthropic_provider, _openai_messages_to_anthropic
from app.providers.ollama_client import ollama_provider
from app.providers.registry import provider_registry
from app.services.cost_estimator import estimate_cost, get_model_pricing


def test_provider_registry_contains_new_providers():
    """Verify Ollama and Anthropic are registered in the global provider_registry."""
    providers = provider_registry.list_providers()
    assert "ollama" in providers
    assert "anthropic" in providers


def test_ollama_supports_model():
    """Verify OllamaProvider correctly matches open-source models."""
    assert ollama_provider.supports_model("llama3")
    assert ollama_provider.supports_model("llama3.1")
    assert ollama_provider.supports_model("mistral")
    assert ollama_provider.supports_model("qwen")
    assert ollama_provider.supports_model("ollama/my-custom-model")
    assert not ollama_provider.supports_model("gpt-4o")


def test_anthropic_supports_model():
    """Verify AnthropicProvider matches Claude 3 family models."""
    assert anthropic_provider.supports_model("claude-3-5-sonnet")
    assert anthropic_provider.supports_model("claude-3-5-haiku")
    assert anthropic_provider.supports_model("claude-3-opus")
    assert not anthropic_provider.supports_model("gemini-2.0-flash")


def test_openai_to_anthropic_message_conversion():
    """Test system prompt extraction for Anthropic format."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write hello world in python."},
    ]
    sys_prompt, anthropic_msgs = _openai_messages_to_anthropic(messages)
    assert sys_prompt == "You are a helpful coding assistant."
    assert len(anthropic_msgs) == 1
    assert anthropic_msgs[0] == {"role": "user", "content": "Write hello world in python."}


def test_cost_estimator_ollama_zero_cost():
    """Verify local Ollama model calls are estimated at $0.00 cost."""
    cost = estimate_cost("ollama/llama3", tokens_input=1000, tokens_output=500)
    assert cost == 0.0

    in_price, out_price = get_model_pricing("llama3")
    assert in_price == 0.0
    assert out_price == 0.0


def test_cost_estimator_anthropic_pricing():
    """Verify Anthropic Claude model pricing."""
    cost = estimate_cost("claude-3-5-sonnet", tokens_input=1_000_000, tokens_output=1_000_000)
    assert cost == 18.0  # $3.00 input + $15.00 output
