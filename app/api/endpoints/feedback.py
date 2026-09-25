"""
Feedback & Router Training Endpoints.

POST /api/v1/feedback                  — Submit per-request quality feedback
GET  /api/v1/feedback/{request_id}     — Get all feedback for a request
POST /api/v1/router/train              — Manually trigger router retraining
GET  /api/v1/router/training-history   — View past training runs
GET  /api/v1/analytics/quality-cost    — Quality-per-dollar tradeoff report
"""

import logging
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.db.models import RequestFeedback, RequestLog, RouterTrainingRun
from app.db.session import get_db

logger = logging.getLogger("optillm.api.feedback")

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Per-request quality feedback payload."""
    rating: Literal[-1, 0, 1] = Field(
        ...,
        description="1 = good response, -1 = bad response, 0 = neutral"
    )
    issue: Optional[Literal["wrong", "slow", "incomplete", "hallucinated", "off-topic"]] = Field(
        None,
        description="Structured issue category (optional)"
    )
    note: Optional[str] = Field(
        None,
        max_length=1000,
        description="Free-form feedback note (optional)"
    )
    user_id: Optional[str] = Field(None, description="User submitting the feedback")
    team_id: Optional[str] = Field(None, description="Team attribution")


class FeedbackResponse(BaseModel):
    id: int
    request_id: Optional[str]
    rating: int
    issue: Optional[str]
    note: Optional[str]
    created_at: str
    training_label_queued: bool


class RouterTrainRequest(BaseModel):
    force: bool = Field(
        False,
        description="Force training even if below minimum sample threshold"
    )
    min_samples: int = Field(
        30,
        ge=5,
        description="Minimum training samples required"
    )


# ── Feedback Endpoints ─────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    summary="Submit per-request quality feedback",
)
def submit_feedback(
    payload: FeedbackRequest,
    request_id: str = Query(..., description="The chatcmpl-xxx request ID to give feedback on"),
    db: Session = Depends(get_db),
):
    """
    Submit structured quality feedback for a completed request.

    Feedback is stored in `request_feedback` and automatically generates
    a training label for the AI Router:
      - rating=-1 on a cheap model → complexity bumped to higher tier
      - rating=1  → complexity label confirmed as correct

    This closes the router learning loop without any manual intervention.
    """
    # Look up the request log by request_id (stored in prompt_snippet prefix as chatcmpl-xxx)
    log = (
        db.query(RequestLog)
        .filter(RequestLog.prompt_snippet.isnot(None))
        .order_by(RequestLog.timestamp.desc())
        .limit(500)  # Search recent requests only for performance
        .all()
    )
    matched_log = None
    for r in log:
        # request_id is stored as first part of prompt_snippet if we can match it
        # Fallback: match by recency if request_id is the last N
        if request_id and str(r.id) == request_id:
            matched_log = r
            break

    request_log_id = matched_log.id if matched_log else -1

    # Store feedback
    feedback = RequestFeedback(
        request_log_id=request_log_id,
        request_id=request_id,
        rating=payload.rating,
        issue=payload.issue,
        note=payload.note,
        user_id=payload.user_id,
        team_id=payload.team_id,
    )
    db.add(feedback)

    # Auto-generate a router training label from the feedback
    training_label_queued = False
    if matched_log and matched_log.prompt_snippet:
        _queue_training_label_from_feedback(
            db=db,
            log=matched_log,
            rating=payload.rating,
            issue=payload.issue,
        )
        training_label_queued = True

    db.commit()
    db.refresh(feedback)

    return FeedbackResponse(
        id=feedback.id,
        request_id=request_id,
        rating=feedback.rating,
        issue=feedback.issue,
        note=feedback.note,
        created_at=feedback.created_at.isoformat() if feedback.created_at else "",
        training_label_queued=training_label_queued,
    )


def _queue_training_label_from_feedback(
    db: Session,
    log: RequestLog,
    rating: int,
    issue: Optional[str],
) -> None:
    """
    Convert user feedback into a router training label.

    Logic:
      - rating=-1 on a cheap model → this request was too complex for the routed model
        → label as complexity one tier higher
      - rating=1  → routing was correct, confirm label
      - rating=0  → no label generated (neutral)
    """
    from app.db.models import RouterTrainingLabel

    if rating == 0:
        return  # Neutral — no label

    cheap_models = {"gpt-4o-mini", "gemini-2.0-flash", "claude-3-5-haiku-20241022"}
    high_models = {"gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"}

    model_used = (log.model_used or "").lower()
    is_cheap_model = any(m in model_used for m in ["mini", "flash", "haiku", "8b"])
    is_high_model = any(m in model_used for m in ["gpt-4o", "sonnet", "opus", "1.5-pro"])

    if rating == -1 and is_cheap_model:
        # Bad response from cheap model → was mis-routed → should be "high" complexity
        complexity_label = "high"
        confidence = 0.85
    elif rating == 1 and is_cheap_model:
        # Good response from cheap model → routing was correct → "low" complexity
        complexity_label = "low"
        confidence = 0.90
    elif rating == -1 and is_high_model:
        # Bad response even from premium model → quality issue, not routing
        # Still label as "high" since we needed the expensive model
        complexity_label = "high"
        confidence = 0.60
    elif rating == 1 and is_high_model:
        complexity_label = "high"
        confidence = 0.80
    else:
        complexity_label = "medium"
        confidence = 0.70

    label = RouterTrainingLabel(
        request_log_id=log.id,
        prompt_snippet=log.prompt_snippet,
        model_requested=log.model_requested,
        complexity_label=complexity_label,
        label_source="feedback",
        confidence=confidence,
    )
    db.add(label)


@router.get(
    "/feedback/request/{request_id}",
    tags=["Feedback"],
    summary="Get all feedback for a request",
)
def get_feedback_for_request(
    request_id: str,
    db: Session = Depends(get_db),
):
    """Returns all feedback submitted for a specific request ID."""
    feedback_list = (
        db.query(RequestFeedback)
        .filter(RequestFeedback.request_id == request_id)
        .order_by(RequestFeedback.created_at.desc())
        .all()
    )
    return {
        "request_id": request_id,
        "feedback": [
            {
                "id": f.id,
                "rating": f.rating,
                "issue": f.issue,
                "note": f.note,
                "user_id": f.user_id,
                "created_at": f.created_at.isoformat() if f.created_at else None,
            }
            for f in feedback_list
        ],
        "count": len(feedback_list),
    }


@router.get(
    "/feedback/stats",
    tags=["Feedback"],
    summary="Aggregate feedback statistics",
)
def get_feedback_stats(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    db: Session = Depends(get_db),
):
    """Returns aggregate feedback stats: thumbs up/down counts, issue breakdown, etc."""
    from sqlalchemy import func
    from datetime import timedelta

    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(RequestFeedback)
        .filter(RequestFeedback.created_at >= since)
        .all()
    )

    total = len(rows)
    positive = sum(1 for r in rows if r.rating == 1)
    negative = sum(1 for r in rows if r.rating == -1)
    neutral = sum(1 for r in rows if r.rating == 0)

    issue_counts: dict = {}
    for r in rows:
        if r.issue:
            issue_counts[r.issue] = issue_counts.get(r.issue, 0) + 1

    return {
        "period_days": days,
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "satisfaction_rate": round(positive / total, 4) if total > 0 else None,
        "issue_breakdown": issue_counts,
    }


# ── Router Training Endpoints ──────────────────────────────────────────────────

@router.post(
    "/router/train",
    tags=["Router"],
    summary="Manually trigger AI Router retraining",
)
def trigger_router_training(
    payload: RouterTrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Manually triggers an AI Router retraining run.

    The training pipeline:
    1. Collects labeled data from DB (human feedback + inferred labels)
    2. Trains a new RandomForest classifier
    3. Evaluates on a 20% held-out set
    4. Hot-swaps if accuracy improves over current model

    Training runs in the background. Check /router/training-history for results.
    """
    from app.engine.router_trainer import train_router

    def _run_training():
        try:
            result = train_router(db, force=payload.force, min_samples=payload.min_samples)
            logger.info("Manual training complete: %s", result)
        except Exception as e:
            logger.error("Manual training failed: %s", e)

    background_tasks.add_task(_run_training)

    return {
        "status": "training_queued",
        "message": (
            "Router retraining started in background. "
            "Check GET /api/v1/router/training-history for results."
        ),
        "force": payload.force,
    }


@router.get(
    "/router/training-history",
    tags=["Router"],
    summary="View AI Router training run history",
)
def get_training_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Returns the history of router retraining runs with accuracy metrics."""
    try:
        runs = (
            db.query(RouterTrainingRun)
            .order_by(RouterTrainingRun.started_at.desc())
            .limit(limit)
            .all()
        )
        return {
            "runs": [
                {
                    "id": r.id,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "samples_trained": r.samples_trained,
                    "samples_evaluated": r.samples_evaluated,
                    "current_model_accuracy": r.current_accuracy,
                    "new_model_accuracy": r.new_accuracy,
                    "model_swapped": r.model_swapped,
                    "training_time_ms": r.training_time_ms,
                }
                for r in runs
            ],
            "total_runs": len(runs),
        }
    except Exception as e:
        return {"runs": [], "error": str(e), "note": "Run alembic upgrade head to create training tables."}


# ── Quality-Cost Analytics Endpoint ───────────────────────────────────────────

@router.get(
    "/analytics/quality-cost",
    tags=["Analytics"],
    summary="Quality-per-dollar tradeoff analysis",
)
def get_quality_cost_tradeoff(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    min_requests: int = Query(5, ge=1, description="Min evaluated requests per model"),
    db: Session = Depends(get_db),
):
    """
    Returns quality-cost tradeoff analysis per model.

    Answers: "Which model gives the best quality per dollar for MY specific workload?"

    Requires:
      - EVAL_ENABLED=true (heuristic eval on every request)
      - At least `min_requests` evaluated requests per model

    Enable EVAL_LLM_ENABLED=true for significantly more accurate quality scores.
    """
    from app.services.analytics import get_quality_cost_tradeoff as _get_tradeoff

    def _parse_dt(s):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.strptime(s, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(400, f"Invalid date format: {s}")

    return _get_tradeoff(
        db,
        start_date=_parse_dt(start_date),
        end_date=_parse_dt(end_date),
        tag=tag,
        min_requests=min_requests,
    )
