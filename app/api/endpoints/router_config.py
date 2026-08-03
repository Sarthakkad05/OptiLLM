"""
Router Configuration, Explainability, and Training Endpoints
Exposes APIs for inspecting and updating routing config, generating explainability reports,
and retraining the AI router ML classifier.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db

from app.engine.ai_router import get_ai_router
from app.engine.router import (
    explain_routing,
    get_routing_config,
    update_routing_config,
)
from app.services.router_exporter import (
    export_historical_routing_data,
    generate_synthetic_training_dataset,
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


class RouterTrainRequest(BaseModel):
    use_synthetic_fallback: bool = Field(
        True, description="Fallback to synthetic dataset if database logs are sparse (< 10 logs)"
    )
    max_logs: int = Field(1000, ge=10, le=10000, description="Max historical database logs to use")


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


@router.post("/train", summary="Train/retrain AI router classifier")
def train_router(
    payload: RouterTrainRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Retrain the AI Router ML classifier using database request logs or synthetic seed dataset.
    """
    exported = export_historical_routing_data(db, max_rows=payload.max_logs)

    features = exported["features"]
    labels = exported["labels"]

    if len(features) < 10 and payload.use_synthetic_fallback:
        synth_features, synth_labels = generate_synthetic_training_dataset()
        features.extend(synth_features)
        labels.extend(synth_labels)

    if not features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient training data available to train AI router.",
        )

    try:
        ai_router = get_ai_router()
        train_result = ai_router.train(features, labels)
        train_result["total_samples"] = len(labels)
        return train_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to train AI router: {str(e)}",
        )
