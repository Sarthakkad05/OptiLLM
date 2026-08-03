"""
Pydantic Schemas for Phase 11 Evaluation Engine
Defines request and response objects for evaluation, A/B model comparison, and quality analytics.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DimensionScores(BaseModel):
    correctness: float = Field(..., ge=0.0, le=1.0, description="Factual accuracy score")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Prompt relevance score")
    completeness: float = Field(..., ge=0.0, le=1.0, description="Completeness & depth score")
    conciseness: float = Field(..., ge=0.0, le=1.0, description="Conciseness score")
    safety: float = Field(..., ge=0.0, le=1.0, description="Content safety score")


class EvaluationRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="Prompt messages context")
    response_text: str = Field(..., description="LLM response text to evaluate")
    reference_context: Optional[str] = Field(None, description="Optional ground truth/reference text")
    model_used: Optional[str] = Field("gpt-4o", description="Model that produced the response")
    cost_usd: Optional[float] = Field(0.0, ge=0.0, description="USD cost incurred for the request")


class EvaluationResponse(BaseModel):
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Composite weighted quality score")
    dimension_scores: DimensionScores = Field(..., description="Breakdown across 5 dimensions")
    hallucination_score: float = Field(..., ge=0.0, le=1.0, description="Hallucination risk score (0=clean, 1=hallucinated)")
    hallucination_explanation: Optional[str] = Field(None, description="Explanation of hallucination findings")
    efficiency_score: float = Field(..., ge=0.0, description="Quality / cost efficiency score")
    efficiency_rating: str = Field(..., description="Efficiency tier: ultra_high | high | moderate | low")
    eval_mode: str = Field(..., description="Evaluation mode used: llm_judge | heuristic")


class ABCompareRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="Prompt messages")
    model_a: str = Field(..., description="First model identifier (e.g. gpt-4o)")
    response_a: str = Field(..., description="Response from model A")
    cost_a: float = Field(0.0, ge=0.0, description="Cost incurred by model A")
    model_b: str = Field(..., description="Second model identifier (e.g. gemini-2.0-flash)")
    response_b: str = Field(..., description="Response from model B")
    cost_b: float = Field(0.0, ge=0.0, description="Cost incurred by model B")
    reference_context: Optional[str] = Field(None, description="Optional reference context")


class ABCompareResponse(BaseModel):
    model_a: str
    eval_a: EvaluationResponse
    model_b: str
    eval_b: EvaluationResponse
    winner: str = Field(..., description="Winning option: model_a | model_b | tie")
    winner_model: str = Field(..., description="Name of the winning model")
    efficiency_ratio: float = Field(..., description="Efficiency ratio (eval_a / eval_b)")
    recommendation_reason: str = Field(..., description="Explanation of A/B test winner decision")


class QualityAnalyticsResponse(BaseModel):
    total_evaluated_requests: int
    avg_quality_score: float
    avg_hallucination_score: float
    avg_efficiency_score: float
    dimension_averages: Dict[str, float]
    model_quality_breakdown: List[Dict[str, Any]]
    quality_over_time: List[Dict[str, Any]]
