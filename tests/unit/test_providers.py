from app.providers.anthropic_client import _openai_messages_to_anthropic
from app.providers.registry import provider_registry


def test_provider_registry_contains_defaults():
    providers = provider_registry.list_providers()
    assert "openai" in providers
    assert "gemini" in providers
    assert "anthropic" in providers


def test_provider_registry_model_lookup():
    openai_lookup = provider_registry.get_for_model("gpt-4o")
    assert openai_lookup is not None
    assert openai_lookup.name == "openai"

    gemini_lookup = provider_registry.get_for_model("gemini-2.0-flash")
    assert gemini_lookup is not None
    assert gemini_lookup.name == "gemini"

    anthropic_lookup = provider_registry.get_for_model("claude-3-5-sonnet-20241022")
    assert anthropic_lookup is not None
    assert anthropic_lookup.name == "anthropic"


def test_anthropic_message_conversion():
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello Claude!"},
    ]
    system_prompt, converted = _openai_messages_to_anthropic(messages)

    assert system_prompt == "You are a helpful AI assistant."
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == "Hello Claude!"
