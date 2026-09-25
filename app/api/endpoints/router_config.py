"""
Router Configuration & Explainability Endpoints
Exposes APIs for inspecting and updating routing config, and generating
explainability reports.

Retraining the AI router lives at POST /api/v1/router/train in
app/api/endpoints/feedback.py, which calls the real evaluate-then-hotswap
pipeline in app/engine/router_trainer.py. An earlier, simpler /router/train
route used to live here too, calling AIRouter.train() directly with no
train/eval split and no hot-swap check — it silently shadowed the safe one
below it in the route table (FastAPI matches routes in registration order),
so the safety gate the differentiator is supposed to provide was never
actually reachable via the API. Removed rather than fixed in place, since
feedback.py's version is the complete, correct implementation.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.engine.router import (
    explain_routing,
    get_routing_config,
    update_routing_config,
)

router = APIRouter(prefix="/router", tags=["Router"])


class RouterConfigUpdateRequest(BaseModel):
    low_model: Optional[str] = Field(None, description="Model for LOW complexity tasks")
    medium_model: Optional[str] = Field(None, description="Model for MEDIUM complexity tasks")
    routing_mode: Optional[str] = Field(None, description="Routing mode: rule_based | ai | shadow")
    confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence threshold for AI router (0.0 to 1.0)"
    )


class RouterExplainRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="Prompt messages to analyze")
    model: str = Field("gpt-4o", description="Requested LLM model")


@router.get("/config", summary="Get routing configuration")
def get_config() -> Dict[str, Any]:
    """Get current routing table, strategy, active mode, and thresholds."""
    return get_routing_config()


@router.post("/config", summary="Update routing configuration")
def update_config(payload: RouterConfigUpdateRequest) -> Dict[str, Any]:
    """Update routing table target models, routing mode, or AI confidence threshold."""
    try:
        updated = update_routing_config(
            low_model=payload.low_model,
            medium_model=payload.medium_model,
            routing_mode=payload.routing_mode,
            confidence_threshold=payload.confidence_threshold,
        )
        return updated
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/explain", summary="Generate routing explainability report")
def explain(payload: RouterExplainRequest) -> Dict[str, Any]:
    """
    Generate complete explainability breakdown for a prompt and model,
    including rule-based scores, AI probabilities, feature vector, and fallback state.
    """
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty."
        )
    return explain_routing(payload.messages, payload.model)
