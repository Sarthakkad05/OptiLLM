"""
Pydantic schemas for the OpenAI-compatible chat completions endpoint.
Mirrors the OpenAI API contract so existing clients work with zero changes.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


# ── Inbound Request ──────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str


class OptiLLMConfig(BaseModel):
    """Optional block clients can include to control OptiLLM behaviour."""
    bypass_cache: bool = False
    bypass_compression: bool = False
    bypass_routing: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Target model (e.g. gpt-4o, gemini-1.5-flash)")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False         # Streaming not supported in MVP
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



class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo
    optillm_metadata: OptiLLMMetadata
