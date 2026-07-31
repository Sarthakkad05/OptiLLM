"""
optillm_client/client.py
The OptiLLM drop-in client — wraps openai.OpenAI transparently.
"""

import openai
from optillm_client.types import OptiLLMMetadata


class _EnrichedCompletion:
    """
    Wraps a standard ChatCompletion object and adds `.optillm_metadata`.
    All original attributes (choices, usage, id, model, etc.) are preserved.
    """
    def __init__(self, raw_response, metadata: OptiLLMMetadata):
        self._raw = raw_response
        self.optillm_metadata = metadata

    def __getattr__(self, name: str):
        return getattr(self._raw, name)

    def __repr__(self) -> str:
        return f"<EnrichedCompletion model={self._raw.model!r} | {self.optillm_metadata!r}>"


class _OptiLLMCompletions:
    """Intercepts chat.completions.create() calls to parse OptiLLM metadata."""

    def __init__(self, original_completions, gateway_url: str):
        self._completions = original_completions
        self._gateway_url = gateway_url

    def create(self, **kwargs) -> _EnrichedCompletion:
        """
        Drop-in replacement for openai.chat.completions.create().
        Returns an enriched response with `.optillm_metadata` attached.
        """
        response = self._completions.create(**kwargs)

        # Extract optillm_metadata from the raw API response dict
        # openai SDK stores the raw body in response.model_extra or _raw_response
        metadata = OptiLLMMetadata()
        try:
            raw_dict = response.model_dump() if hasattr(response, "model_dump") else {}
            meta_dict = raw_dict.get("optillm_metadata") or {}
            if meta_dict:
                metadata = OptiLLMMetadata.from_dict(meta_dict)
        except Exception:
            pass  # Metadata enrichment is best-effort — never break the call

        return _EnrichedCompletion(response, metadata)

    async def acreate(self, **kwargs) -> _EnrichedCompletion:
        """Async version of create()."""
        response = await self._completions.acreate(**kwargs)
        metadata = OptiLLMMetadata()
        try:
            raw_dict = response.model_dump() if hasattr(response, "model_dump") else {}
            meta_dict = raw_dict.get("optillm_metadata") or {}
            if meta_dict:
                metadata = OptiLLMMetadata.from_dict(meta_dict)
        except Exception:
            pass
        return _EnrichedCompletion(response, metadata)


class _OptiLLMChat:
    """Wraps the openai.chat namespace."""
    def __init__(self, original_chat, gateway_url: str):
        self.completions = _OptiLLMCompletions(original_chat.completions, gateway_url)


class OptiLLM:
    """
    Drop-in replacement for openai.OpenAI that routes through the OptiLLM gateway.

    Every API call is semantically cached, context-compressed, and intelligently
    routed to the most cost-effective model — transparently.

    Example:
        # Before (direct OpenAI)
        from openai import OpenAI
        client = OpenAI(api_key="sk-...")

        # After (OptiLLM — zero other changes needed)
        from optillm_client import OptiLLM
        client = OptiLLM(api_key="sk-...", gateway_url="http://localhost:8000")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response.optillm_metadata)
    """

    def __init__(
        self,
        api_key: str = None,
        gateway_url: str = "http://localhost:8000",
        **openai_kwargs,
    ):
        """
        Args:
            api_key: Your OpenAI API key. Passed through to the gateway.
            gateway_url: URL of the running OptiLLM gateway.
            **openai_kwargs: Any other kwargs accepted by openai.OpenAI
                             (e.g. organization, timeout, max_retries).
        """
        self._gateway_url = gateway_url.rstrip("/")

        # Point the underlying OpenAI client at the OptiLLM gateway
        self._client = openai.OpenAI(
            api_key=api_key or "optillm",   # key is passed through by the gateway
            base_url=f"{self._gateway_url}/v1",
            **openai_kwargs,
        )

        self.chat = _OptiLLMChat(self._client.chat, self._gateway_url)

    def health(self) -> dict:
        """Check if the OptiLLM gateway is reachable."""
        import requests
        resp = requests.get(f"{self._gateway_url}/health", timeout=5)
        return resp.json()

    @property
    def gateway_url(self) -> str:
        return self._gateway_url

    def __repr__(self) -> str:
        return f"OptiLLM(gateway={self._gateway_url!r})"
