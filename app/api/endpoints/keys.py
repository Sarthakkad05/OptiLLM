"""
API Key Management Endpoints.
POST   /v1/keys              — Create a new API key
GET    /v1/keys              — List all API keys
POST   /v1/keys/{key_id}/rotate — Rotate an API key (generate new secret, invalidate old)
DELETE /v1/keys/{key_id}     — Revoke an API key

Keys are stored hashed (SHA-256) and never returned in plaintext after creation or rotation.
"""

import hashlib
import json
import logging
import secrets
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, Integer, String, Text, text
from sqlalchemy.orm import Session

from app.db.base_class import Base
from app.db.session import get_db

logger = logging.getLogger("optillm.api.keys")

router = APIRouter()


# ── Key DB Model ───────────────────────────────────────────────────────────────

class APIKeyRecord(Base):
    """Stores API key metadata. Raw key is never stored — only SHA-256 hash."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(32), unique=True, nullable=False, index=True)
    key_hash = Column(String(64), nullable=False, unique=True)
    name = Column(String(200), nullable=True)
    project = Column(String(200), nullable=True)
    scopes = Column(Text, nullable=True)  # JSON list of scopes
    created_at = Column(Integer, nullable=False, default=lambda: int(time.time()))
    rotated_at = Column(Integer, nullable=True)  # Unix timestamp of last key rotation
    expires_at = Column(Integer, nullable=True)  # Unix timestamp, None = no expiry
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(Integer, nullable=True)
    # Per-key rate limit overrides (None = use global settings)
    rpm_limit = Column(Integer, nullable=True)  # Requests per minute
    tpm_limit = Column(Integer, nullable=True)  # Tokens per minute


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _ensure_table(db: Session):
    """Ensure api_keys table exists and has all required columns (idempotent)."""
    try:
        db.execute(text("SELECT 1 FROM api_keys LIMIT 1"))
        # Ensure rotated_at column exists on older tables
        try:
            db.execute(text("SELECT rotated_at FROM api_keys LIMIT 1"))
        except Exception:
            try:
                db.rollback()
                db.execute(text("ALTER TABLE api_keys ADD COLUMN rotated_at INTEGER"))
                db.commit()
            except Exception:
                db.rollback()
    except Exception:
        db.rollback()
        from app.db.session import engine
        Base.metadata.create_all(bind=engine, tables=[APIKeyRecord.__table__])


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    name: Optional[str] = None
    project: Optional[str] = None
    scopes: Optional[List[str]] = None
    expires_in_days: Optional[int] = None
    rpm_limit: Optional[int] = None  # Requests per minute override (None = global default)
    tpm_limit: Optional[int] = None  # Tokens per minute override (None = global default)


class CreateKeyResponse(BaseModel):
    key_id: str
    api_key: str  # Only returned once at creation — never again
    name: Optional[str]
    project: Optional[str]
    scopes: Optional[List[str]]
    created_at: int
    expires_at: Optional[int]
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None
    message: str = "Store this key safely — it will not be shown again."


class RotateKeyRequest(BaseModel):
    expires_in_days: Optional[int] = None


class RotateKeyResponse(BaseModel):
    key_id: str
    api_key: str  # Newly generated key — only returned once
    name: Optional[str]
    project: Optional[str]
    scopes: Optional[List[str]]
    rotated_at: int
    expires_at: Optional[int]
    message: str = "Key rotated successfully. Store this new key safely — old key has been invalidated."


class KeyInfo(BaseModel):
    key_id: str
    name: Optional[str]
    project: Optional[str]
    scopes: Optional[List[str]]
    created_at: int
    rotated_at: Optional[int] = None
    expires_at: Optional[int]
    is_active: bool
    last_used_at: Optional[int]
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None


class KeysListResponse(BaseModel):
    keys: List[KeyInfo]
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/v1/keys",
    response_model=CreateKeyResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["API Keys"],
    summary="Create API Key",
    description="Creates a new API key. The raw key is returned ONCE and never stored.",
)
def create_key(
    request: CreateKeyRequest,
    db: Session = Depends(get_db),
) -> CreateKeyResponse:
    """Creates and returns a new OptiLLM API key."""
    _ensure_table(db)

    # Generate a secure random key with sk-optillm- prefix
    raw_key = f"sk-optillm-{secrets.token_urlsafe(32)}"
    key_id = secrets.token_hex(16)
    key_hash = _hash_key(raw_key)

    expires_at = None
    if request.expires_in_days:
        expires_at = int(time.time()) + (request.expires_in_days * 86400)

    record = APIKeyRecord(
        key_id=key_id,
        key_hash=key_hash,
        name=request.name,
        project=request.project,
        scopes=json.dumps(request.scopes or ["chat"]),
        created_at=int(time.time()),
        expires_at=expires_at,
        is_active=True,
        rpm_limit=request.rpm_limit,
        tpm_limit=request.tpm_limit,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    logger.info("Created new API key: key_id=%s name=%s project=%s", key_id, request.name, request.project)

    return CreateKeyResponse(
        key_id=key_id,
        api_key=raw_key,
        name=request.name,
        project=request.project,
        scopes=request.scopes or ["chat"],
        created_at=record.created_at,
        expires_at=expires_at,
        rpm_limit=request.rpm_limit,
        tpm_limit=request.tpm_limit,
    )


@router.post(
    "/v1/keys/{key_id}/rotate",
    response_model=RotateKeyResponse,
    tags=["API Keys"],
    summary="Rotate API Key",
    description="Generates a new secret for the key and immediately invalidates the previous secret.",
)
def rotate_key(
    key_id: str,
    request: Optional[RotateKeyRequest] = None,
    db: Session = Depends(get_db),
) -> RotateKeyResponse:
    """Rotates an existing active API key."""
    _ensure_table(db)

    record = db.query(APIKeyRecord).filter(APIKeyRecord.key_id == key_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Key '{key_id}' not found.",
        )

    if not record.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot rotate revoked or inactive key '{key_id}'.",
        )

    # Generate new raw key and replace hash
    new_raw_key = f"sk-optillm-{secrets.token_urlsafe(32)}"
    now = int(time.time())
    record.key_hash = _hash_key(new_raw_key)
    record.rotated_at = now

    if request and request.expires_in_days:
        record.expires_at = now + (request.expires_in_days * 86400)

    db.commit()
    db.refresh(record)

    scopes_list = json.loads(record.scopes) if record.scopes else ["chat"]
    logger.info("Rotated API key: key_id=%s", key_id)

    return RotateKeyResponse(
        key_id=key_id,
        api_key=new_raw_key,
        name=record.name,
        project=record.project,
        scopes=scopes_list,
        rotated_at=record.rotated_at,
        expires_at=record.expires_at,
    )


@router.get(
    "/v1/keys",
    response_model=KeysListResponse,
    tags=["API Keys"],
    summary="List API Keys",
)
def list_keys(db: Session = Depends(get_db)) -> KeysListResponse:
    """Lists all registered API keys (raw keys are never returned)."""
    _ensure_table(db)

    records = db.query(APIKeyRecord).order_by(APIKeyRecord.created_at.desc()).all()
    keys = [
        KeyInfo(
            key_id=r.key_id,
            name=r.name,
            project=r.project,
            scopes=json.loads(r.scopes) if r.scopes else ["chat"],
            created_at=r.created_at,
            rotated_at=getattr(r, "rotated_at", None),
            expires_at=r.expires_at,
            is_active=r.is_active,
            last_used_at=r.last_used_at,
            rpm_limit=r.rpm_limit,
            tpm_limit=r.tpm_limit,
        )
        for r in records
    ]
    return KeysListResponse(keys=keys, total=len(keys))


@router.delete(
    "/v1/keys/{key_id}",
    tags=["API Keys"],
    summary="Revoke API Key",
)
def revoke_key(key_id: str, db: Session = Depends(get_db)):
    """Revokes (deactivates) an API key by ID."""
    _ensure_table(db)

    record = db.query(APIKeyRecord).filter(APIKeyRecord.key_id == key_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Key '{key_id}' not found.")

    record.is_active = False
    db.commit()

    logger.info("Revoked API key: key_id=%s", key_id)
    return {"key_id": key_id, "status": "revoked", "message": "API key has been deactivated."}
