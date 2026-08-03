"""
BasePlugin Abstract Interface
Defines the enterprise plugin lifecycle hooks: initialize, pre_process, post_process, and on_error.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePlugin(ABC):
    """
    Abstract Base Class for enterprise gateway plugins.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Called once when plugin is loaded into memory."""
        pass

    @abstractmethod
    def pre_process(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoked before LLM provider request dispatch.
        Allows modifying request parameters, prompt messages, or guardrail validation.
        """
        return request_payload

    @abstractmethod
    def post_process(self, response_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoked after LLM provider response returned.
        Allows inspecting/transforming final completion response.
        """
        return response_payload

    def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Invoked when gateway error occurs."""
        pass
