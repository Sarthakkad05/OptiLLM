"""
Unit tests for OptiLLM Python SDK and CLI interface.
"""

from optillm_client.client import OptiLLMClient


def test_optillm_client_initialization():
    client = OptiLLMClient(base_url="http://localhost:8000", api_key="sk-test-key")
    assert client.base_url == "http://localhost:8000"
    assert client._headers()["Authorization"] == "Bearer sk-test-key"

    # Verify all expected namespaces are present
    assert hasattr(client, "chat")
    assert hasattr(client, "embeddings")
    assert hasattr(client, "keys")
    assert hasattr(client, "teams")
    assert hasattr(client, "users")
    assert hasattr(client, "guardrails")
    assert hasattr(client, "analytics")
    assert hasattr(client, "feedback")
    assert hasattr(client, "prompts")
    assert hasattr(client, "router")


def test_async_optillm_client_initialization():
    from optillm_client.async_client import AsyncOptiLLMClient
    async_client = AsyncOptiLLMClient(base_url="http://localhost:8000", api_key="sk-async-key")
    assert async_client.base_url == "http://localhost:8000"
    assert async_client._headers()["Authorization"] == "Bearer sk-async-key"

    # Verify full namespace parity
    assert hasattr(async_client, "chat")
    assert hasattr(async_client, "embeddings")
    assert hasattr(async_client, "keys")
    assert hasattr(async_client, "teams")
    assert hasattr(async_client, "users")
    assert hasattr(async_client, "guardrails")
    assert hasattr(async_client, "analytics")
    assert hasattr(async_client, "feedback")
    assert hasattr(async_client, "prompts")
    assert hasattr(async_client, "router")


def test_cli_argument_parsing(monkeypatch):
    from optillm_client.cli import main
    import sys

    # Test status command runs without error (mocking cmd_status)
    called = []
    monkeypatch.setattr("optillm_client.cli.cmd_status", lambda args: called.append("status"))
    monkeypatch.setattr(sys, "argv", ["optillm", "status"])
    main()
    assert "status" in called

    # Test chat argument parsing
    chat_called = []
    monkeypatch.setattr("optillm_client.cli.cmd_chat", lambda args: chat_called.append(args.prompt))
    monkeypatch.setattr(sys, "argv", ["optillm", "chat", "Hello world!", "--model", "gpt-4o"])
    main()
    assert "Hello world!" in chat_called
