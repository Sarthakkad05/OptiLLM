"""
Pydantic schemas for the OpenAI-compatible chat completions endpoint.
Mirrors the OpenAI API contract so existing clients work with zero changes.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ── Inbound Request ──────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


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
    stream: Optional[bool] = False  # Streaming not supported in MVP
    optillm: Optional[OptiLLMConfig] = OptiLLMConfig()


# ── Outbound Response ─────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str


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
