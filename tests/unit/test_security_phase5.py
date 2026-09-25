"""
Comprehensive Unit & Integration Test Suite for Phase 5 (Security Overhaul):
- 5.1 Presidio PII Detection Engine & Regex Fallback
- 5.2 API Key Rotation, Expiry, Revocation & Scopes
- 5.3 Secrets Management Backends (Env, File, Injection)
- 5.4 Token Bucket Rate Limiter, Global Limiter & AutoBlocklist
"""

import os
import tempfile
import time
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.auth import require_scopes, verify_api_key
from app.core.config import settings
from app.core.rate_limiter import (
    AutoBlocklist,
    TokenBucketRateLimiter,
    auto_blocklist,
    check_rate_limit,
    token_bucket_limiter,
)
from app.core.secrets_loader import (
    EnvSecretsBackend,
    FileSecretsBackend,
    get_secrets_backend,
    load_provider_secrets,
)
from app.db.session import SessionLocal
from app.api.endpoints.keys import (
    APIKeyRecord,
    CreateKeyRequest,
    RotateKeyRequest,
    create_key,
    revoke_key,
    rotate_key,
)
from app.security.pii import PIIDetector, get_pii_detector


# ── 5.1 Presidio PII Engine Tests ─────────────────────────────────────────────

def test_presidio_pii_detection_and_redaction():
    detector = get_pii_detector()
    sample_text = (
        "Hello, my name is John Doe. My email is john.doe@example.com, "
        "phone number is 555-123-4567, and my secret key is sk_test_FAKE_KEY_FOR_UNIT_TESTING_ONLY."
    )
    result = detector.redact(sample_text)

    assert result["pii_found"] is True
    assert len(result["detected_types"]) > 0
    # The text should no longer contain the raw sensitive items
    assert "john.doe@example.com" not in result["redacted_text"]
    assert "sk_test_FAKE_KEY_FOR_UNIT_TESTING_ONLY" not in result["redacted_text"]


def test_pii_detector_regex_fallback():
    detector = PIIDetector(enable_presidio=False)
    assert detector.get_engine_type() == "regex"

    sample_text = "Contact me at alice@test.org or SSN 123-45-6789."
    result = detector.redact(sample_text)

    assert result["pii_found"] is True
    assert "EMAIL" in result["detected_types"]
    assert "SSN" in result["detected_types"]
    assert "[REDACTED_EMAIL]" in result["redacted_text"]
    assert "[REDACTED_SSN]" in result["redacted_text"]


def test_pii_detector_analyze_detailed():
    detector = get_pii_detector()
    sample_text = "Send credentials to bob@example.com"
    entities = detector.analyze(sample_text)

    assert len(entities) >= 1
    email_entities = [e for e in entities if "EMAIL" in e["entity_type"].upper()]
    assert len(email_entities) >= 1
    assert "bob@example.com" in email_entities[0]["text"]


# ── 5.2 API Key Lifecycle & Rotation Tests ─────────────────────────────────────

def test_api_key_creation_and_hash_storage():
    with SessionLocal() as db:
        req = CreateKeyRequest(name="agent-prod", scopes=["chat", "analytics"])
        res = create_key(req, db=db)

        assert res.key_id is not None
        assert res.api_key.startswith("sk-optillm-")
        assert res.scopes == ["chat", "analytics"]

        # Confirm hash is stored in DB, raw key is NOT stored
        record = db.query(APIKeyRecord).filter(APIKeyRecord.key_id == res.key_id).first()
        assert record is not None
        assert record.key_hash != res.api_key
        assert len(record.key_hash) == 64  # SHA-256


def test_api_key_rotation_invalidates_old_key():
    with SessionLocal() as db:
        # Create key
        created = create_key(CreateKeyRequest(name="rot-key", scopes=["chat"]), db=db)
        old_raw = created.api_key

        # Rotate key
        rotated = rotate_key(created.key_id, request=RotateKeyRequest(expires_in_days=30), db=db)
        new_raw = rotated.api_key

        assert new_raw.startswith("sk-optillm-")
        assert new_raw != old_raw
        assert rotated.rotated_at is not None
        assert rotated.expires_at is not None


@pytest.mark.asyncio
async def test_auth_verification_and_expiry():
    orig_auth = settings.API_KEY_AUTH_ENABLED
    settings.API_KEY_AUTH_ENABLED = True
    try:
        with SessionLocal() as db:
            # 1. Active key
            active = create_key(CreateKeyRequest(name="auth-test", scopes=["chat"]), db=db)
            creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=active.api_key)
            verified = await verify_api_key(creds)
            assert verified == active.api_key

            # 2. Expired key
            expired = create_key(CreateKeyRequest(name="expired-key", scopes=["chat"]), db=db)
            # Set expires_at in the past
            rec = db.query(APIKeyRecord).filter(APIKeyRecord.key_id == expired.key_id).first()
            rec.expires_at = int(time.time()) - 100
            db.commit()

            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(
                    HTTPAuthorizationCredentials(scheme="Bearer", credentials=expired.api_key)
                )
            assert exc_info.value.status_code == 401
            assert "expired" in exc_info.value.detail.lower()

            # 3. Revoked key
            revoke_key(active.key_id, db=db)
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(creds)
            assert exc_info.value.status_code == 401
            assert "revoked" in exc_info.value.detail.lower()
    finally:
        settings.API_KEY_AUTH_ENABLED = orig_auth


@pytest.mark.asyncio
async def test_scope_authorization():
    orig_auth = settings.API_KEY_AUTH_ENABLED
    settings.API_KEY_AUTH_ENABLED = True
    try:
        with SessionLocal() as db:
            key = create_key(CreateKeyRequest(name="chat-only", scopes=["chat"]), db=db)

            # Chat scope allowed
            chat_dep = require_scopes("chat")
            res = await chat_dep(token=key.api_key)
            assert res == key.api_key

            # Admin scope rejected with 403
            admin_dep = require_scopes("admin")
            with pytest.raises(HTTPException) as exc_info:
                await admin_dep(token=key.api_key)
            assert exc_info.value.status_code == 403
            assert "Insufficient permissions" in exc_info.value.detail
    finally:
        settings.API_KEY_AUTH_ENABLED = orig_auth


# ── 5.3 Secrets Management Backend Tests ──────────────────────────────────────

def test_env_secrets_backend():
    os.environ["OPTILLM_TEST_SECRET"] = "super-secret-token"
    backend = EnvSecretsBackend()
    assert backend.get_secret("OPTILLM_TEST_SECRET") == "super-secret-token"
    assert backend.get_secret("NON_EXISTENT_KEY", default="fallback") == "fallback"


def test_file_secrets_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        secret_path = os.path.join(tmpdir, "OPENAI_API_KEY")
        with open(secret_path, "w", encoding="utf-8") as f:
            f.write("sk-file-provider-token\n")

        backend = FileSecretsBackend(secrets_dir=tmpdir)
        retrieved = backend.get_secret("OPENAI_API_KEY")
        assert retrieved == "sk-file-provider-token"


def test_load_provider_secrets():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "ANTHROPIC_API_KEY"), "w") as f:
            f.write("sk-ant-test-token")

        backend = FileSecretsBackend(secrets_dir=tmpdir)
        with patch("app.core.secrets_loader.get_secrets_backend", return_value=backend):
            class DummySettings:
                ANTHROPIC_API_KEY = ""
            target = DummySettings()
            loaded = load_provider_secrets(target=target)
            assert "ANTHROPIC_API_KEY" in loaded
            assert target.ANTHROPIC_API_KEY == "sk-ant-test-token"


# ── 5.4 Token Bucket Rate Limiter Tests ────────────────────────────────────────

def test_token_bucket_rate_limiter_burst_and_refill():
    limiter = TokenBucketRateLimiter(requests_per_minute=120, burst_factor=1.0)
    client = "client-tb-1"

    # Consume all 120 burst tokens
    for i in range(120):
        allowed, retry, rem = limiter.consume(client, 1.0)
        assert allowed is True, f"Failed at iteration {i}"

    # 121st should fail
    allowed, retry, rem = limiter.consume(client, 1.0)
    assert allowed is False
    assert retry > 0

    # Simulate 1 second elapsed -> should refill ~2 tokens (120 RPM / 60s = 2 tokens/sec)
    tokens, last_time = limiter._buckets[client]
    limiter._buckets[client] = (tokens, last_time - 1.0)
    allowed, retry, rem = limiter.consume(client, 1.0)
    assert allowed is True


def test_auto_blocklist_behavior():
    blocklist = AutoBlocklist(threshold=3, window_seconds=10.0, block_duration=60.0)
    actor = "192.168.1.100"

    assert blocklist.is_blocked(actor)[0] is False
    blocklist.record_violation(actor)
    blocklist.record_violation(actor)
    assert blocklist.is_blocked(actor)[0] is False

    # 3rd breach triggers auto-block
    blocked_now = blocklist.record_violation(actor)
    assert blocked_now is True
    is_blocked, remaining = blocklist.is_blocked(actor)
    assert is_blocked is True
    assert remaining > 0

    # Unblock
    blocklist.unblock(actor)
    assert blocklist.is_blocked(actor)[0] is False


@pytest.mark.asyncio
async def test_check_rate_limit_blocked_client():
    mock_request = MagicMock()
    mock_request.headers.get.return_value = None
    mock_request.client.host = "10.0.0.99"

    # Place client on blocklist
    auto_blocklist.record_violation("10.0.0.99")
    auto_blocklist.record_violation("10.0.0.99")
    auto_blocklist.record_violation("10.0.0.99")
    auto_blocklist.record_violation("10.0.0.99")
    auto_blocklist.record_violation("10.0.0.99")  # 5th violation triggers block

    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(mock_request)

    assert exc_info.value.status_code == 429
    assert "temporarily blocked" in exc_info.value.detail

    # Clean up
    auto_blocklist.unblock("10.0.0.99")
