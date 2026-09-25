"""
Azure OpenAI Provider Adapter.
Routes requests to Azure OpenAI endpoints using deployment-name-based routing.
Azure uses the same OpenAI response format but requires different auth and URL structure.
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from app.core.config import settings
from app.providers.base import BaseProvider

logger = logging.getLogger("optillm.provider.azure")

_PLACEHOLDER_KEYS = {"", None}

# Maps canonical OpenAI model names to Azure deployment name defaults.
# Users can override AZURE_DEPLOYMENT_MAP via env: "gpt-4o=my-gpt4o-deployment,gpt-4o-mini=my-mini"
_DEFAULT_DEPLOYMENT_MAP = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-35-turbo",  # Azure uses gpt-35-turbo (no dot)
}


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI Provider Adapter."""

    @property
    def name(self) -> str:
        return "azure"

    def is_available(self) -> bool:
        return (
            settings.AZURE_OPENAI_API_KEY not in _PLACEHOLDER_KEYS
            and settings.AZURE_OPENAI_ENDPOINT not in _PLACEHOLDER_KEYS
        )

    def supports_model(self, model: str) -> bool:
        return model.lower() in _DEFAULT_DEPLOYMENT_MAP or "gpt" in model.lower()

    def _get_deployment(self, model: str) -> str:
        """Maps model name to Azure deployment name."""
        return _DEFAULT_DEPLOYMENT_MAP.get(model, model)

    def _chat_url(self, deployment: str) -> str:
        endpoint = settings.AZURE_OPENAI_ENDPOINT.rstrip("/")
        api_version = settings.AZURE_OPENAI_API_VERSION
        return f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"

    def _headers(self) -> Dict[str, str]:
        return {
            "api-key": settings.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

    async def call(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError(
                "Azure OpenAI credentials not set. Configure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
            )

        deployment = self._get_deployment(model)
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info(
            "Calling Azure OpenAI | deployment=%s | model=%s | messages=%d",
            deployment, model, len(messages),
        )
        start = time.time()

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                self._chat_url(deployment),
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()

        elapsed_ms = int((time.time() - start) * 1000)
        data = response.json()
        usage = data.get("usage", {})

        return {
            "id": data.get("id", f"azure-{uuid.uuid4().hex[:8]}"),
            "content": data["choices"][0]["message"]["content"],
            "model": model,
            "tokens_input": usage.get("prompt_tokens", 0),
            "tokens_output": usage.get("completion_tokens", 0),
            "finish_reason": data["choices"][0].get("finish_reason", "stop"),
            "latency_ms": elapsed_ms,
            "provider": "azure",
        }

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        if not self.is_available():
            raise ValueError("Azure OpenAI credentials not set.")

        deployment = self._get_deployment(model)
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST", self._chat_url(deployment),
                json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            if content:
                                yield content
                        except Exception:
                            continue


azure_provider = AzureOpenAIProvider()
