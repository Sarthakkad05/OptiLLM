"""
Unit tests for OptiLLM Python SDK and CLI interface.
"""

from optillm_client.client import OptiLLMClient


def test_optillm_client_initialization():
    client = OptiLLMClient(base_url="http://localhost:8000", api_key="sk-test-key")
    assert client.base_url == "http://localhost:8000"
    assert client._headers["Authorization"] == "Bearer sk-test-key"
