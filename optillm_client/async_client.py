"""
OptiLLM Python SDK — Async Client
Full async mirror of client.py using httpx.AsyncClient.

Usage:
    import asyncio
    from optillm_client import AsyncOptiLLMClient

    async def main():
        async with AsyncOptiLLMClient(api_key="sk-...", base_url="http://localhost:8000") as client:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            print(response.content)

    asyncio.run(main())
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from optillm_client.client import (
    ChatCompletion,
    Choice,
    CreateEmbeddingResponse,
    EmbeddingData,
    Message,
    Usage,
)


class _AsyncCompletionsNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncIterator[str]]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        if stream:
            return self._client._stream("/v1/chat/completions", payload)

        data = await self._client._post("/v1/chat/completions", payload)
        choices = [
            Choice(
                index=c["index"],
                message=Message(role=c["message"]["role"], content=c["message"]["content"]),
                finish_reason=c.get("finish_reason", "stop"),
            )
            for c in data.get("choices", [])
        ]
        usage_data = data.get("usage", {})
        return ChatCompletion(
            id=data["id"],
            object=data.get("object", "chat.completion"),
            created=data.get("created", 0),
            model=data.get("model", "unknown"),
            choices=choices,
            usage=Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            optillm_metadata=data.get("optillm_metadata"),
        )


class _AsyncChatNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self.completions = _AsyncCompletionsNamespace(client)


class _AsyncEmbeddingsNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def create(
        self,
        input: Union[str, List[str]],
        model: str = "text-embedding-3-small",
    ) -> CreateEmbeddingResponse:
        data = await self._client._post("/v1/embeddings", {"input": input, "model": model})
        return CreateEmbeddingResponse(
            object=data.get("object", "list"),
            data=[
                EmbeddingData(object=d.get("object", "embedding"), embedding=d["embedding"], index=d["index"])
                for d in data.get("data", [])
            ],
            model=data.get("model", model),
            usage=data.get("usage", {}),
        )


class _AsyncKeysNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def create(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self._client._post("/v1/keys", {
            "name": name,
            "project": project,
            "scopes": scopes,
            "expires_in_days": expires_in_days,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
        })

    async def list(self) -> Dict[str, Any]:
        return await self._client._get("/v1/keys")

    async def delete(self, key_id: str) -> Dict[str, Any]:
        return await self._client._delete(f"/v1/keys/{key_id}")


class _AsyncTeamsNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def create(
        self,
        name: str,
        org_id: Optional[str] = None,
        daily_budget_usd: float = 100.0,
        monthly_budget_usd: float = 1000.0,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self._client._post("/v1/teams", {
            "name": name,
            "org_id": org_id,
            "daily_budget_usd": daily_budget_usd,
            "monthly_budget_usd": monthly_budget_usd,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
        })

    async def list(self) -> Dict[str, Any]:
        return await self._client._get("/v1/teams")

    async def get(self, team_id: str) -> Dict[str, Any]:
        return await self._client._get(f"/v1/teams/{team_id}")

    async def delete(self, team_id: str) -> Dict[str, Any]:
        return await self._client._delete(f"/v1/teams/{team_id}")


class _AsyncUsersNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def create(
        self,
        email: Optional[str] = None,
        team_id: Optional[str] = None,
        role: str = "developer",
        daily_budget_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        return await self._client._post("/v1/users", {
            "email": email,
            "team_id": team_id,
            "role": role,
            "daily_budget_usd": daily_budget_usd,
        })

    async def list(self, team_id: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if team_id:
            params["team_id"] = team_id
        return await self._client._get("/v1/users", params=params)

    async def get(self, user_id: str) -> Dict[str, Any]:
        return await self._client._get(f"/v1/users/{user_id}")

    async def delete(self, user_id: str) -> Dict[str, Any]:
        return await self._client._delete(f"/v1/users/{user_id}")


class _AsyncGuardrailsNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def list(self) -> Dict[str, Any]:
        return await self._client._get("/api/v1/guardrails")

    async def set_enabled(self, name: str, enabled: bool) -> Dict[str, Any]:
        return await self._client._patch(f"/api/v1/guardrails/{name}", {"enabled": enabled})

    async def test(self, messages: List[Dict]) -> Dict[str, Any]:
        return await self._client._post("/api/v1/guardrails/test", {"messages": messages})


class _AsyncAnalyticsNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def summary(self, days: int = 7) -> Dict[str, Any]:
        return await self._client._get("/api/v1/analytics", params={"days": days})

    async def quality_cost_tradeoff(self) -> Dict[str, Any]:
        return await self._client._get("/api/v1/analytics/quality-cost-tradeoff")


class _AsyncFeedbackNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def submit(
        self,
        request_id: str,
        rating: int,
        issue: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client._post(f"/api/v1/feedback/{request_id}", {
            "rating": rating,
            "issue": issue,
            "notes": notes,
        })

    async def stats(self) -> Dict[str, Any]:
        return await self._client._get("/api/v1/feedback/stats")


class _AsyncPromptsNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def list(self) -> List[Dict[str, Any]]:
        return await self._client._get("/api/v1/prompts")

    async def render(
        self, name: str, variables: Dict[str, Any], version: str = "v1"
    ) -> Dict[str, Any]:
        return await self._client._post("/api/v1/prompts/render", {
            "name": name,
            "version": version,
            "variables": variables,
        })

    async def create_version(
        self,
        name: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
        commit_message: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._client._post(f"/api/v1/prompts/{name}/versions", {
            "system_template": system_template,
            "user_template": user_template,
            "description": description,
            "commit_message": commit_message,
            "version": version,
        })

    async def history(self, name: str) -> List[Dict[str, Any]]:
        return await self._client._get(f"/api/v1/prompts/{name}/versions")

    async def rollback(self, name: str, version: str) -> Dict[str, Any]:
        return await self._client._post(f"/api/v1/prompts/{name}/rollback/{version}", {})

    async def compare(
        self, name: str, version_a: str, version_b: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        return await self._client._post(f"/api/v1/prompts/{name}/compare", {
            "version_a": version_a,
            "version_b": version_b,
            "variables": variables,
        })


class _AsyncRouterNamespace:
    def __init__(self, client: "AsyncOptiLLMClient"):
        self._client = client

    async def train(self) -> Dict[str, Any]:
        return await self._client._post("/api/v1/router/train", {})

    async def history(self) -> Dict[str, Any]:
        return await self._client._get("/api/v1/router/training-history")


class AsyncOptiLLMClient:
    """
    OptiLLM Python SDK — async client.
    Use as async context manager:
        async with AsyncOptiLLMClient(...) as client:
            response = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("OPTILLM_API_KEY", "sk-optillm-dev-key")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None

        self.chat = _AsyncChatNamespace(self)
        self.embeddings = _AsyncEmbeddingsNamespace(self)
        self.keys = _AsyncKeysNamespace(self)
        self.teams = _AsyncTeamsNamespace(self)
        self.users = _AsyncUsersNamespace(self)
        self.guardrails = _AsyncGuardrailsNamespace(self)
        self.analytics = _AsyncAnalyticsNamespace(self)
        self.feedback = _AsyncFeedbackNamespace(self)
        self.prompts = _AsyncPromptsNamespace(self)
        self.router = _AsyncRouterNamespace(self)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def _post(self, path: str, payload: Dict) -> Dict:
        response = await self._get_client().post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        response = await self._get_client().get(
            f"{self.base_url}{path}",
            params=params,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _delete(self, path: str) -> Dict:
        response = await self._get_client().delete(
            f"{self.base_url}{path}",
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _patch(self, path: str, payload: Dict) -> Dict:
        response = await self._get_client().patch(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _stream(self, path: str, payload: Dict) -> AsyncIterator[str]:
        """Yields content chunks from an async streaming chat completion."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content")
                            )
                            if content:
                                yield content
                        except Exception:
                            continue

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
