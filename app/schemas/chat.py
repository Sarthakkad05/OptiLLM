"""
Pydantic schemas for the OpenAI-compatible chat completions endpoint.
Mirrors the OpenAI API contract so existing clients work with zero changes.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# ── Inbound Request ──────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str = "user"
    content: Optional[str] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "allow"}


class OptiLLMConfig(BaseModel):
    """Optional block clients can include to control OptiLLM behaviour."""

    bypass_cache: bool = False
    bypass_compression: bool = False
    bypass_routing: bool = False
    cache_threshold: Optional[float] = None
    cache_namespace: Optional[str] = "default"
    ttl_seconds: Optional[int] = None
    compression_mode: Optional[Literal["smart", "aggressive", "minimal"]] = "smart"


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Target model (e.g. gpt-4o, gemini-1.5-flash)")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    optillm: Optional[OptiLLMConfig] = OptiLLMConfig()

    model_config = {"extra": "allow"}


# ── Outbound Response ─────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"


class OptiLLMMetadata(BaseModel):
    """Extra optimisation metadata appended to every response."""

    cache_hit: bool
    compressed: bool
    routed: bool
    model_requested: str
    model_used: str
    latency_ms: int
    cost_usd: float
    savings_usd: float
    tokens_saved: int
    routing_reason: Optional[str] = None
    complexity: Optional[str] = None

    model_config = {"protected_namespaces": ()}


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo
    optillm_metadata: OptiLLMMetadata
