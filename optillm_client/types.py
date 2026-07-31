"""
optillm_client/types.py
OptiLLM response metadata types.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptiLLMMetadata:
    """
    Enriched metadata returned by the OptiLLM gateway on every response.
    Attached to the standard OpenAI response object as `.optillm_metadata`.
    """
    cache_hit: bool = False            # True if served from semantic cache (free!)
    compressed: bool = False           # True if context was compressed before LLM call
    routed: bool = False               # True if request was routed to a cheaper model

    model_requested: str = ""          # What you asked for (e.g. "gpt-4o")
    model_used: str = ""               # What was actually used (e.g. "gpt-4o-mini")
    provider: str = ""                 # "openai" | "gemini" | "mock" | "cache"

    latency_ms: int = 0                # Total gateway latency in milliseconds
    cost_usd: float = 0.0             # Actual cost incurred (0 on cache hit)
    savings_usd: float = 0.0          # Money saved (cache + compression + routing)
    tokens_saved: int = 0             # Tokens saved by compression

    routing_reason: Optional[str] = None   # Human-readable explanation of routing decision
    complexity: Optional[str] = None       # "low" | "medium" | "high"

    @classmethod
    def from_dict(cls, data: dict) -> "OptiLLMMetadata":
        """Parse from the optillm_metadata dict in the API response."""
        return cls(
            cache_hit=data.get("cache_hit", False),
            compressed=data.get("compressed", False),
            routed=data.get("routed", False),
            model_requested=data.get("model_requested", ""),
            model_used=data.get("model_used", ""),
            provider=data.get("provider", ""),
            latency_ms=data.get("latency_ms", 0),
            cost_usd=data.get("cost_usd", 0.0),
            savings_usd=data.get("savings_usd", 0.0),
            tokens_saved=data.get("tokens_saved", 0),
            routing_reason=data.get("routing_reason"),
            complexity=data.get("complexity"),
        )

    def __repr__(self) -> str:
        status = []
        if self.cache_hit:
            status.append("⚡ CACHE HIT")
        if self.compressed:
            status.append("🗜️ COMPRESSED")
        if self.routed:
            status.append(f"🔀 ROUTED → {self.model_used}")
        status_str = " | ".join(status) if status else "direct LLM call"
        return (
            f"OptiLLMMetadata({status_str} | "
            f"latency={self.latency_ms}ms | "
            f"cost=${self.cost_usd:.6f} | "
            f"saved=${self.savings_usd:.6f})"
        )
