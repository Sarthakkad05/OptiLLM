from typing import Dict, List, Optional

from pydantic import BaseModel


class RequestLogOut(BaseModel):
    """Schema for a single request log row returned to the client."""

    id: int
    timestamp: Optional[str] = None
    model_requested: Optional[str] = None
    model_used: str
    provider: str
    tag: Optional[str] = None
    cache_hit: bool
    compressed: bool
    routed: bool
    tokens_input: int
    tokens_output: int
    tokens_saved: int
    cost_usd: float
    savings_usd: float
    latency_ms: int
    prompt_snippet: Optional[str] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}


class DashboardSummary(BaseModel):
    """Top-level KPI summary."""

    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    total_cost_usd: float
    total_savings_usd: float
    total_tokens_input: int
    total_tokens_output: int
    total_tokens_saved: int
    avg_latency_ms: float


class CostOverTime(BaseModel):
    date: str
    cost_usd: float
    savings_usd: float
    requests: int


class ModelDistribution(BaseModel):
    model: str
    requests: int


class AnalyticsResponse(BaseModel):
    """Full analytics payload returned by GET /analytics."""

    summary: DashboardSummary
    cost_over_time: List[CostOverTime]
    model_distribution: List[ModelDistribution]
    recent_requests: List[RequestLogOut]

    model_config = {"protected_namespaces": ()}


# ── Phase 4 Additions ──────────────────────────────────────────────────────────


class LatencyPercentilesResponse(BaseModel):
    """Latency percentiles overall and breakdown by provider."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_requests: int
    per_provider: Dict[str, Dict[str, float]]


class TokenTrendPoint(BaseModel):
    date: str
    tokens_input: int
    tokens_output: int
    tokens_saved: int


class TokenTrendsResponse(BaseModel):
    trends: List[TokenTrendPoint]


class ProviderAnalyticsItem(BaseModel):
    provider: str
    request_count: int
    avg_latency_ms: float
    cache_hit_rate: float
    cost_usd: float
    savings_usd: float


class ProviderAnalyticsResponse(BaseModel):
    providers: List[ProviderAnalyticsItem]


class SavingsBreakdownResponse(BaseModel):
    cache_savings_usd: float
    compression_savings_usd: float
    routing_savings_usd: float
    total_savings_usd: float
