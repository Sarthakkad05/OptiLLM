"""
Secrets Management Backend.
Provides pluggable secrets backends for production environments:
  - 'env'   : Standard environment variables (default)
  - 'file'  : Docker / Kubernetes file-based secrets (e.g. /run/secrets/<secret_name>)
  - 'aws'   : AWS Secrets Manager (boto3)
  - 'vault' : HashiCorp Vault (hvac or HTTP API)

Eliminates hardcoded credentials and enables secure, dynamic retrieval of LLM provider keys.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger("optillm.core.secrets")


class SecretsBackend(ABC):
    """Abstract base class for all secrets backends."""

    @abstractmethod
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a single secret by key."""
        pass

    @abstractmethod
    def load_all_secrets(self) -> Dict[str, str]:
        """Load all secrets managed by this backend into a dictionary."""
        pass


class EnvSecretsBackend(SecretsBackend):
    """Default backend reading from OS environment variables."""

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        val = os.getenv(key)
        return val if val is not None and val != "" else default

    def load_all_secrets(self) -> Dict[str, str]:
        return dict(os.environ)


class FileSecretsBackend(SecretsBackend):
    """
    Reads secrets from filesystem files, standard for Docker swarm and Kubernetes mounts
    (e.g., /run/secrets/OPENAI_API_KEY).
    """

    def __init__(self, secrets_dir: Optional[str] = None):
        self.secrets_dir = Path(secrets_dir or settings.SECRETS_DIR)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Try exact filename match, lowercase match, or uppercase match
        candidates = [
            self.secrets_dir / key,
            self.secrets_dir / key.lower(),
            self.secrets_dir / key.upper(),
        ]
        for path in candidates:
            if path.is_file():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except Exception as exc:
                    logger.warning("Error reading secret file %s: %s", path, exc)

        # Fallback to env if file not present
        env_val = os.getenv(key)
        return env_val if env_val is not None and env_val != "" else default

    def load_all_secrets(self) -> Dict[str, str]:
        secrets = {}
        if self.secrets_dir.is_dir():
            for p in self.secrets_dir.iterdir():
                if p.is_file():
                    try:
                        secrets[p.name] = p.read_text(encoding="utf-8").strip()
                    except Exception as exc:
                        logger.warning("Failed to read secret file %s: %s", p, exc)
        return secrets


class AWSSecretsBackend(SecretsBackend):
    """
    AWS Secrets Manager backend using boto3.
    Retrieves and caches secrets stored as JSON or string under AWS_SECRET_NAME.
    """

    def __init__(
        self,
        secret_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        self.secret_name = secret_name or settings.AWS_SECRET_NAME
        self.region_name = region_name or settings.AWS_REGION
        self._cached_secrets: Optional[Dict[str, str]] = None

    def _fetch_from_aws(self) -> Dict[str, str]:
        if self._cached_secrets is not None:
            return self._cached_secrets

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.warning("boto3 not installed — AWSSecretsBackend falling back to env")
            return dict(os.environ)

        try:
            client = boto3.client(
                "secretsmanager",
                region_name=self.region_name,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            )
            response = client.get_secret_value(SecretId=self.secret_name)
            secret_string = response.get("SecretString", "")
            if secret_string:
                try:
                    self._cached_secrets = json.loads(secret_string)
                except json.JSONDecodeError:
                    self._cached_secrets = {self.secret_name: secret_string}
            else:
                self._cached_secrets = {}
            logger.info("Successfully loaded secrets from AWS Secrets Manager (%s)", self.secret_name)
            return self._cached_secrets
        except Exception as exc:
            logger.error("AWS Secrets Manager error (%s) — falling back to env", exc)
            return dict(os.environ)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        secrets = self._fetch_from_aws()
        val = secrets.get(key) or os.getenv(key)
        return val if val is not None and val != "" else default

    def load_all_secrets(self) -> Dict[str, str]:
        return self._fetch_from_aws()


class VaultSecretsBackend(SecretsBackend):
    """
    HashiCorp Vault backend.
    Retrieves secrets via hvac client (if installed) or directly via Vault REST API.
    """

    def __init__(
        self,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        secret_path: Optional[str] = None,
    ):
        self.vault_addr = (vault_addr or settings.VAULT_ADDR).rstrip("/")
        self.vault_token = vault_token or settings.VAULT_TOKEN
        self.secret_path = secret_path or settings.VAULT_SECRET_PATH
        self._cached_secrets: Optional[Dict[str, str]] = None

    def _fetch_from_vault(self) -> Dict[str, str]:
        if self._cached_secrets is not None:
            return self._cached_secrets

        if not self.vault_addr or not self.vault_token:
            logger.warning("Vault address or token unconfigured — falling back to env")
            return dict(os.environ)

        # Attempt with hvac if installed
        try:
            import hvac

            client = hvac.Client(url=self.vault_addr, token=self.vault_token)
            if client.is_authenticated():
                # Supports KV v2 reading
                parts = self.secret_path.split("/", 1)
                mount = parts[0]
                path = parts[1] if len(parts) > 1 else ""
                secret_resp = client.secrets.kv.v2.read_secret_version(
                    path=path, mount_point=mount
                )
                data = secret_resp.get("data", {}).get("data", {})
                self._cached_secrets = {k: str(v) for k, v in data.items()}
                logger.info("Successfully fetched secrets from Vault via hvac")
                return self._cached_secrets
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("hvac client failed (%s) — trying Vault REST API", exc)

        # Fallback to direct HTTP request using httpx
        try:
            import httpx

            url = f"{self.vault_addr}/v1/{self.secret_path}"
            headers = {"X-Vault-Token": self.vault_token}
            response = httpx.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                body = response.json()
                data = body.get("data", {}).get("data", body.get("data", {}))
                self._cached_secrets = {k: str(v) for k, v in data.items()}
                logger.info("Successfully fetched secrets from Vault REST API")
                return self._cached_secrets
            else:
                logger.warning("Vault REST API returned status %d", response.status_code)
        except Exception as exc:
            logger.error("Vault REST error: %s", exc)

        return dict(os.environ)

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        secrets = self._fetch_from_vault()
        val = secrets.get(key) or os.getenv(key)
        return val if val is not None and val != "" else default

    def load_all_secrets(self) -> Dict[str, str]:
        return self._fetch_from_vault()


_backend_instance: Optional[SecretsBackend] = None


def get_secrets_backend(backend_type: Optional[str] = None) -> SecretsBackend:
    """Returns singleton instance of the configured secrets backend."""
    global _backend_instance
    b_type = (backend_type or settings.SECRETS_BACKEND).lower()

    if _backend_instance is not None and backend_type is None:
        return _backend_instance

    if b_type == "file":
        backend = FileSecretsBackend()
    elif b_type == "aws":
        backend = AWSSecretsBackend()
    elif b_type == "vault":
        backend = VaultSecretsBackend()
    else:
        backend = EnvSecretsBackend()

    if backend_type is None:
        _backend_instance = backend
    return backend


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve secret using active secrets backend."""
    return get_secrets_backend().get_secret(key, default=default)


def load_provider_secrets(target=None) -> Dict[str, str]:
    """
    Loads known provider keys from the active backend and injects them into
    the target object (defaults to settings singleton).
    """
    target = target or settings
    backend = get_secrets_backend()

    known_keys = [
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "MISTRAL_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    loaded = {}
    for key in known_keys:
        val = backend.get_secret(key)
        if val:
            setattr(target, key, val)
            loaded[key] = f"***{val[-4:]}" if len(val) > 4 else "***"

    if loaded:
        logger.info("Loaded provider secrets from backend '%s': %s", settings.SECRETS_BACKEND, list(loaded.keys()))
    return loaded
