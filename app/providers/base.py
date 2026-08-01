"""
Abstract Base Provider.
Defines the standard interface contract for all LLM providers in OptiLLM.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional


class BaseProvider(ABC):
    """
    Abstract interface for LLM provider adapters.
    All providers (OpenAI, Gemini, Anthropic, etc.) must inherit from this class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns provider unique identifier (e.g. 'openai', 'gemini', 'anthropic')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if provider has valid API key / credentials configured."""
        pass

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Returns True if this provider supports target model name."""
        pass

    @abstractmethod
    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes a non-streaming completion request.
        Returns normalized response dictionary.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Executes a streaming completion request.
        Yields content text delta strings.
        """
        pass
