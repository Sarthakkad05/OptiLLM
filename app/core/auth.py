"""
API Key Authentication and Authorization Dependency.
Validates Bearer token headers against static OPTILLM_API_KEYS and dynamic database APIKeyRecords.
Enforces expiration (expires_at), active status (is_active), and scope-based permissions.
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

logger = logging.getLogger("optillm.core.auth")

security_scheme = HTTPBearer(auto_error=False)


def _hash_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def get_key_details_from_db(token: str) -> Optional[Dict[str, Any]]:
    """
    Looks up an API key record in the database by its SHA-256 hash.
    Returns key details dictionary or None if not found or DB unavailable.
    """
    try:
        from app.api.endpoints.keys import APIKeyRecord
        from app.db.session import SessionLocal

        key_hash = _hash_key(token)
        with SessionLocal() as db:
            record = db.query(APIKeyRecord).filter(APIKeyRecord.key_hash == key_hash).first()
            if not record:
                return None

            scopes = []
            if record.scopes:
                try:
                    scopes = json.loads(record.scopes)
                except Exception:
                    scopes = ["chat"]

            return {
                "key_id": record.key_id,
                "name": record.name,
                "project": record.project,
                "scopes": scopes,
                "is_active": record.is_active,
                "expires_at": record.expires_at,
                "created_at": record.created_at,
                "rpm_limit": record.rpm_limit,
                "tpm_limit": record.tpm_limit,
            }
    except Exception as exc:
        logger.warning("Error fetching API key from DB (%s)", exc)
        return None


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
) -> str:
    """
    Validates Bearer API Key from request headers.
    Checks:
      1. If API_KEY_AUTH_ENABLED is False, bypasses authentication (local dev).
      2. If token matches static OPTILLM_API_KEYS (bootstrap keys).
      3. If token exists in DB APIKeyRecord, is active, and has not expired.
    Updates last_used_at on successful DB auth.
    """
    if not settings.API_KEY_AUTH_ENABLED:
        return "anonymous"

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()

    # 1. Check static bootstrap keys
    allowed_keys = [
        k.strip() for k in settings.OPTILLM_API_KEYS.split(",") if k.strip()
    ]
    if token in allowed_keys:
        return token

    # 2. Check dynamic database keys
    try:
        from app.api.endpoints.keys import APIKeyRecord
        from app.db.session import SessionLocal

        key_hash = _hash_key(token)
        with SessionLocal() as db:
            record = db.query(APIKeyRecord).filter(APIKeyRecord.key_hash == key_hash).first()

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API Key provided.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if not record.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API Key has been revoked or deactivated.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            now = int(time.time())
            if record.expires_at is not None and now > record.expires_at:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"API Key expired at {record.expires_at}.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Record usage timestamp
            record.last_used_at = now
            db.commit()

            return token

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DB error during API key verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service temporarily unavailable.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_scopes(*required_scopes: str):
    """
    FastAPI dependency enforcing that the calling API key possesses at least one of
    the required scopes (e.g. 'chat', 'admin', 'analytics', 'prompts').
    Root bootstrap keys and dev mode bypass scope restrictions.
    """
    async def scope_checker(
        token: str = Depends(verify_api_key),
    ) -> str:
        # Dev mode bypass
        if token == "anonymous" or not settings.API_KEY_AUTH_ENABLED:
            return token

        # Bootstrap admin keys have full privileges
        allowed_keys = [
            k.strip() for k in settings.OPTILLM_API_KEYS.split(",") if k.strip()
        ]
        if token in allowed_keys:
            return token

        # Check DB scopes
        key_details = get_key_details_from_db(token)
        if not key_details:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        key_scopes = set(key_details.get("scopes") or ["chat"])

        # Wildcard or admin grant all scopes
        if "*" in key_scopes or "admin" in key_scopes:
            return token

        # Check if any required scope is present
        if not any(req in key_scopes for req in required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: key requires one of {list(required_scopes)}, has {list(key_scopes)}.",
            )

        return token

    return scope_checker
