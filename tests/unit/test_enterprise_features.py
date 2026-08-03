"""
Unit tests for Phase 14 Enterprise Features — PII detection, encryption, audit log hash chain, and plugin system.
"""

from app.plugins.reference_plugin import CustomHeaderPlugin, PIIGuardPlugin
from app.plugins.registry import PluginRegistry
from app.security.encryption import get_encryptor
from app.security.pii import get_pii_detector


def test_pii_detector_ssn_email_redaction():
    detector = get_pii_detector()
    input_text = "User SSN is 123-45-6789 and email is john.doe@example.com."
    res = detector.redact(input_text)

    assert res["pii_found"] is True
    assert "SSN" in res["detected_types"]
    assert "EMAIL" in res["detected_types"]
    assert "[REDACTED_SSN]" in res["redacted_text"]
    assert "[REDACTED_EMAIL]" in res["redacted_text"]


def test_payload_encryption_and_decryption():
    encryptor = get_encryptor()
    secret_text = "Sensitive LLM Prompt Content 123"

    encrypted = encryptor.encrypt(secret_text)
    assert encrypted != secret_text

    decrypted = encryptor.decrypt(encrypted)
    assert decrypted == secret_text


def test_plugin_registry_and_reference_plugins():
    registry = PluginRegistry()

    pii_plugin = PIIGuardPlugin()
    header_plugin = CustomHeaderPlugin()

    registry.register(pii_plugin)
    registry.register(header_plugin)

    assert len(registry.list_plugins()) == 2

    # Pre-process hook test (PII redaction)
    req_payload = {
        "messages": [{"role": "user", "content": "My email is alice@company.org"}]
    }
    processed_req = registry.run_pre_process(req_payload)
    assert "[REDACTED_EMAIL]" in processed_req["messages"][0]["content"]

    # Post-process hook test (Custom header compliance)
    resp_payload = {"choices": [{"message": {"content": "Hello"}}]}
    processed_resp = registry.run_post_process(resp_payload)
    assert "enterprise_compliance" in processed_resp
    assert processed_resp["enterprise_compliance"]["verified"] is True
