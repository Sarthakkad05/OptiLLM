from typing import List, Optional

from pydantic import BaseModel


class RequestLogOut(BaseModel):
    """Schema for a single request log row returned to the client."""

    id: int
    timestamp: Optional[str]
    model_used: str
    provider: str
    cache_hit: bool
    compressed: bool
    routed: bool
    tokens_input: int
    tokens_output: int
    tokens_saved: int
    cost_usd: float
    savings_usd: float
    latency_ms: int
    prompt_snippet: Optional[str]

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
