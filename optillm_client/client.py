"""
OptiLLM Python SDK — Sync Client
Mirrors the OpenAI SDK surface for drop-in compatibility.

Usage:
    from optillm_client import OptiLLMClient

    client = OptiLLMClient(api_key="sk-...", base_url="http://localhost:8000")

    # Chat completions (OpenAI-compatible)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

    # Embeddings
    embeddings = client.embeddings.create(input=["Hello world"], model="text-embedding-3-small")

    # Admin: Keys
    key = client.keys.create(name="my-app", rpm_limit=100)

    # Admin: Teams
    team = client.teams.create(name="Engineering", daily_budget_usd=50.0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

import httpx


# ── Response Models ───────────────────────────────────────────────────────────


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: str = "stop"


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    optillm_metadata: Optional[Dict[str, Any]] = None

    @property
    def content(self) -> str:
        """Shortcut: returns first choice content."""
        return self.choices[0].message.content if self.choices else ""


@dataclass
class EmbeddingData:
    object: str
    embedding: List[float]
    index: int


@dataclass
class CreateEmbeddingResponse:
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


# ── Namespace Classes ─────────────────────────────────────────────────────────


class _CompletionsNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Iterator[str]]:
        """Creates a chat completion. Set stream=True for streaming."""
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

        data = self._client._post("/v1/chat/completions", payload)
        return self._parse_chat_completion(data)

    def _parse_chat_completion(self, data: Dict) -> ChatCompletion:
        choices = [
            Choice(
                index=c["index"],
                message=Message(
                    role=c["message"]["role"],
                    content=c["message"]["content"],
                ),
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


class _ChatNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self.completions = _CompletionsNamespace(client)


class _EmbeddingsNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def create(
        self,
        input: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        **kwargs,
    ) -> CreateEmbeddingResponse:
        """Creates embeddings for the provided text(s)."""
        payload = {"input": input, "model": model, **kwargs}
        data = self._client._post("/v1/embeddings", payload)
        return CreateEmbeddingResponse(
            object=data.get("object", "list"),
            data=[
                EmbeddingData(
                    object=d.get("object", "embedding"),
                    embedding=d["embedding"],
                    index=d["index"],
                )
                for d in data.get("data", [])
            ],
            model=data.get("model", model),
            usage=data.get("usage", {}),
        )


class _KeysNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def create(
        self,
        name: Optional[str] = None,
        project: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Creates a new API key."""
        return self._client._post("/v1/keys", {
            "name": name,
            "project": project,
            "scopes": scopes,
            "expires_in_days": expires_in_days,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
        })

    def list(self) -> Dict[str, Any]:
        return self._client._get("/v1/keys")

    def delete(self, key_id: str) -> Dict[str, Any]:
        return self._client._delete(f"/v1/keys/{key_id}")


class _TeamsNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def create(
        self,
        name: str,
        org_id: Optional[str] = None,
        daily_budget_usd: float = 100.0,
        monthly_budget_usd: float = 1000.0,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._client._post("/v1/teams", {
            "name": name,
            "org_id": org_id,
            "daily_budget_usd": daily_budget_usd,
            "monthly_budget_usd": monthly_budget_usd,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
        })

    def list(self) -> Dict[str, Any]:
        return self._client._get("/v1/teams")

    def get(self, team_id: str) -> Dict[str, Any]:
        return self._client._get(f"/v1/teams/{team_id}")

    def delete(self, team_id: str) -> Dict[str, Any]:
        return self._client._delete(f"/v1/teams/{team_id}")


class _UsersNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def create(
        self,
        email: Optional[str] = None,
        team_id: Optional[str] = None,
        role: str = "developer",
        daily_budget_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self._client._post("/v1/users", {
            "email": email,
            "team_id": team_id,
            "role": role,
            "daily_budget_usd": daily_budget_usd,
        })

    def list(self, team_id: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if team_id:
            params["team_id"] = team_id
        return self._client._get("/v1/users", params=params)

    def get(self, user_id: str) -> Dict[str, Any]:
        return self._client._get(f"/v1/users/{user_id}")

    def delete(self, user_id: str) -> Dict[str, Any]:
        return self._client._delete(f"/v1/users/{user_id}")


class _GuardrailsNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def list(self) -> Dict[str, Any]:
        return self._client._get("/api/v1/guardrails")

    def set_enabled(self, name: str, enabled: bool) -> Dict[str, Any]:
        return self._client._patch(f"/api/v1/guardrails/{name}", {"enabled": enabled})

    def test(self, messages: List[Dict]) -> Dict[str, Any]:
        return self._client._post("/api/v1/guardrails/test", {"messages": messages})


class _AnalyticsNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def summary(self, days: int = 7) -> Dict[str, Any]:
        return self._client._get("/api/v1/analytics", params={"days": days})

    def quality_cost_tradeoff(self) -> Dict[str, Any]:
        return self._client._get("/api/v1/analytics/quality-cost-tradeoff")


class _FeedbackNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def submit(
        self,
        request_id: str,
        rating: int,
        issue: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._client._post(f"/api/v1/feedback/{request_id}", {
            "rating": rating,
            "issue": issue,
            "notes": notes,
        })

    def stats(self) -> Dict[str, Any]:
        return self._client._get("/api/v1/feedback/stats")


class _PromptsNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def list(self) -> List[Dict[str, Any]]:
        return self._client._get("/api/v1/prompts")

    def render(
        self, name: str, variables: Dict[str, Any], version: str = "v1"
    ) -> Dict[str, Any]:
        return self._client._post("/api/v1/prompts/render", {
            "name": name,
            "version": version,
            "variables": variables,
        })

    def create_version(
        self,
        name: str,
        system_template: str,
        user_template: str,
        description: Optional[str] = None,
        commit_message: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._client._post(f"/api/v1/prompts/{name}/versions", {
            "system_template": system_template,
            "user_template": user_template,
            "description": description,
            "commit_message": commit_message,
            "version": version,
        })

    def history(self, name: str) -> List[Dict[str, Any]]:
        return self._client._get(f"/api/v1/prompts/{name}/versions")

    def rollback(self, name: str, version: str) -> Dict[str, Any]:
        return self._client._post(f"/api/v1/prompts/{name}/rollback/{version}", {})

    def compare(
        self, name: str, version_a: str, version_b: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._client._post(f"/api/v1/prompts/{name}/compare", {
            "version_a": version_a,
            "version_b": version_b,
            "variables": variables,
        })


class _RouterNamespace:
    def __init__(self, client: "OptiLLMClient"):
        self._client = client

    def train(self) -> Dict[str, Any]:
        return self._client._post("/api/v1/router/train", {})

    def history(self) -> Dict[str, Any]:
        return self._client._get("/api/v1/router/training-history")


# ── Main Client ───────────────────────────────────────────────────────────────


class OptiLLMClient:
    """
    OptiLLM Python SDK — sync client.
    Mirrors the OpenAI SDK surface for drop-in compatibility.

    Example:
        client = OptiLLMClient(base_url="http://localhost:8000", api_key="sk-...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}]
        )
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

        # Namespaces
        self.chat = _ChatNamespace(self)
        self.embeddings = _EmbeddingsNamespace(self)
        self.keys = _KeysNamespace(self)
        self.teams = _TeamsNamespace(self)
        self.users = _UsersNamespace(self)
        self.guardrails = _GuardrailsNamespace(self)
        self.analytics = _AnalyticsNamespace(self)
        self.feedback = _FeedbackNamespace(self)
        self.prompts = _PromptsNamespace(self)
        self.router = _RouterNamespace(self)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: Dict) -> Dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}{path}",
                params=params,
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def _delete(self, path: str) -> Dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.delete(
                f"{self.base_url}{path}",
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def _patch(self, path: str, payload: Dict) -> Dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.patch(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            return response.json()

    def _stream(self, path: str, payload: Dict) -> Iterator[str]:
        """Yields content chunks from a streaming chat completion."""
        import json
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
