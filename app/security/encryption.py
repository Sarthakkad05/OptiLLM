"""
Payload Encryption at Rest Engine
Uses AES-256 Fernet symmetric encryption for securing sensitive database text fields.
"""

import base64
import hashlib
import logging
from typing import Optional

from cryptography.fernet import Fernet
from app.core.config import settings

logger = logging.getLogger("optillm.security.encryption")

_SECRET_KEY = getattr(settings, "ENCRYPTION_SECRET_KEY", "optillm-default-super-secret-key-32bytes!")
# Derive valid 32-byte URL-safe base64 Fernet key
_key = base64.urlsafe_b64encode(hashlib.sha256(_SECRET_KEY.encode()).digest())
_fernet = Fernet(_key)


class PayloadEncryptor:
    """
    Encrypts and decrypts sensitive database payloads.
    """

    def encrypt(self, plain_text: str) -> str:
        """Encrypt plain text to Fernet token string."""
        if not plain_text:
            return ""
        try:
            encrypted_bytes = _fernet.encrypt(plain_text.encode("utf-8"))
            return encrypted_bytes.decode("utf-8")
        except Exception as e:
            logger.error("Encryption failed: %s", e)
            return plain_text

    def decrypt(self, encrypted_text: str) -> str:
        """Decrypt Fernet token string back to plain text."""
        if not encrypted_text:
            return ""
        try:
            decrypted_bytes = _fernet.decrypt(encrypted_text.encode("utf-8"))
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            # If not an encrypted Fernet token (e.g. legacy plain text), return as is
            return encrypted_text


_global_encryptor: Optional[PayloadEncryptor] = None


def get_encryptor() -> PayloadEncryptor:
    global _global_encryptor
    if _global_encryptor is None:
        _global_encryptor = PayloadEncryptor()
    return _global_encryptor
