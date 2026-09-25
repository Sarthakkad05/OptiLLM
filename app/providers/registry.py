"""
Provider Registry.
Manages registration and lookup of LLM provider adapters.
"""

import logging
from typing import Dict, List, Optional

from app.providers.base import BaseProvider

logger = logging.getLogger("optillm.provider.registry")


class ProviderRegistry:
    """
    Registry pattern for dynamically resolving provider implementations.
    """

    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}

    def register(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        self._providers[provider.name.lower()] = provider
        logger.info("Registered provider adapter: %s", provider.name)

    def get(self, name: str) -> Optional[BaseProvider]:
        """Get provider by identifier name."""
        return self._providers.get(name.lower())

    def get_for_model(self, model: str) -> Optional[BaseProvider]:
        """Find the registered provider adapter that supports the specified model name."""
        for provider in self._providers.values():
            if provider.supports_model(model):
                return provider
        return None

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def list_available_providers(self) -> List[str]:
        """List registered provider names that have valid credentials."""
        return [name for name, p in self._providers.items() if p.is_available()]


# Global singleton provider registry
provider_registry = ProviderRegistry()


def _init_default_providers():
    """Import and register all providers (conditionally based on credentials)."""
    from app.providers.anthropic_client import anthropic_provider
    from app.providers.gemini_client import gemini_provider
    from app.providers.ollama_client import ollama_provider
    from app.providers.openai_client import openai_provider
    from app.providers.groq_client import groq_provider
    from app.providers.azure_openai_client import azure_provider
    from app.providers.mistral_client import mistral_provider
    from app.providers.bedrock_client import bedrock_provider

    # Core providers
    provider_registry.register(openai_provider)
    provider_registry.register(gemini_provider)
    provider_registry.register(anthropic_provider)
    provider_registry.register(ollama_provider)

    # Extended providers (registered always — available() checks credentials at call time)
    provider_registry.register(groq_provider)
    provider_registry.register(azure_provider)
    provider_registry.register(mistral_provider)
    provider_registry.register(bedrock_provider)

    available = provider_registry.list_available_providers()
    logger.info("Available providers: %s", available)


_init_default_providers()
