"""
API Key Authentication Dependency.
Validates Bearer token headers when API_KEY_AUTH_ENABLED is set to True.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

security_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
) -> str:
    """
    Validates Bearer API Key from request headers against allowed OPTILLM_API_KEYS.
    If API_KEY_AUTH_ENABLED is False, authentication is bypassed for local dev.
    """
    if not settings.API_KEY_AUTH_ENABLED:
        return "anonymous"

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    allowed_keys = [
        k.strip() for k in settings.OPTILLM_API_KEYS.split(",") if k.strip()
    ]
    token = credentials.credentials.strip()

    if token not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token
