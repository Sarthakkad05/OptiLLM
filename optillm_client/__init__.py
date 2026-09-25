"""
OptiLLM Client Python SDK & CLI Package

Usage:
    from optillm_client import OptiLLMClient          # sync
    from optillm_client import AsyncOptiLLMClient     # async
"""

from optillm_client.client import OptiLLMClient
from optillm_client.async_client import AsyncOptiLLMClient

__version__ = "1.0.0"
__all__ = ["OptiLLMClient", "AsyncOptiLLMClient"]
