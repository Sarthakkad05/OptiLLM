"""
Evaluation API Endpoints — Response Quality & A/B Model Comparison
Provides endpoints for single-response 5-dimension quality evaluation,
hallucination risk scoring, and side-by-side A/B model comparison.
"""

from fastapi import APIRouter, HTTPException, status

from app.evaluation.hallucination import get_hallucination_detector
from app.evaluation.judge import get_judge
from app.evaluation.scorer import evaluate_cost_efficiency
from app.schemas.evaluation import (
    ABCompareRequest,
    ABCompareResponse,
    DimensionScores,
    EvaluationRequest,
    EvaluationResponse,
)

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.post("", response_model=EvaluationResponse, summary="Evaluate response quality")
def evaluate_response(payload: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluates response text quality across 5 dimensions (Correctness, Relevance, Completeness, Conciseness, Safety),
    detects hallucination risk, and computes cost-vs-quality efficiency rating.
    """
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty."
        )

    judge = get_judge()
    detector = get_hallucination_detector()

    judge_res = judge.evaluate(
        messages=payload.messages,
        response_text=payload.response_text,
        reference_context=payload.reference_context,
    )

    hall_res = detector.detect(
        messages=payload.messages,
        response_text=payload.response_text,
        reference_context=payload.reference_context,
    )

    quality_score = judge_res["quality_score"]
    eff_res = evaluate_cost_efficiency(quality_score, payload.cost_usd or 0.0)

    dim_scores = DimensionScores(**judge_res["dimension_scores"])

    return EvaluationResponse(
        quality_score=quality_score,
        dimension_scores=dim_scores,
        hallucination_score=hall_res["hallucination_score"],
        hallucination_explanation=hall_res["explanation"],
        efficiency_score=eff_res["efficiency_score"],
        efficiency_rating=eff_res["efficiency_rating"],
        eval_mode=judge_res["eval_mode"],
    )


@router.post("/compare", response_model=ABCompareResponse, summary="A/B model response comparison")
def compare_responses(payload: ABCompareRequest) -> ABCompareResponse:
    """
    A/B test two model responses side-by-side on the same prompt.
    Evaluates quality across 5 dimensions, checks hallucination risks, and calculates cost-efficiency ratios
    to declare the winning model.
    """
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty."
        )

    judge = get_judge()
    detector = get_hallucination_detector()

    # Eval Model A
    res_a = judge.evaluate(payload.messages, payload.response_a, payload.reference_context)
    hall_a = detector.detect(payload.messages, payload.response_a, payload.reference_context)
    eff_a = evaluate_cost_efficiency(res_a["quality_score"], payload.cost_a)
    eval_response_a = EvaluationResponse(
        quality_score=res_a["quality_score"],
        dimension_scores=DimensionScores(**res_a["dimension_scores"]),
        hallucination_score=hall_a["hallucination_score"],
        hallucination_explanation=hall_a["explanation"],
        efficiency_score=eff_a["efficiency_score"],
        efficiency_rating=eff_a["efficiency_rating"],
        eval_mode=res_a["eval_mode"],
    )

    # Eval Model B
    res_b = judge.evaluate(payload.messages, payload.response_b, payload.reference_context)
    hall_b = detector.detect(payload.messages, payload.response_b, payload.reference_context)
    eff_b = evaluate_cost_efficiency(res_b["quality_score"], payload.cost_b)
    eval_response_b = EvaluationResponse(
        quality_score=res_b["quality_score"],
        dimension_scores=DimensionScores(**res_b["dimension_scores"]),
        hallucination_score=hall_b["hallucination_score"],
        hallucination_explanation=hall_b["explanation"],
        efficiency_score=eff_b["efficiency_score"],
        efficiency_rating=eff_b["efficiency_rating"],
        eval_mode=res_b["eval_mode"],
    )

    score_diff = res_a["quality_score"] - res_b["quality_score"]
    eff_ratio = round(eff_a["efficiency_score"] / max(0.0001, eff_b["efficiency_score"]), 2)

    if abs(score_diff) < 0.03:
        # Quality is virtually tied, compare efficiency
        if eff_a["efficiency_score"] >= eff_b["efficiency_score"]:
            winner = "model_a"
            winner_model = payload.model_a
            reason = f"Quality score tied (~{res_a['quality_score']:.2f}), but {payload.model_a} is more cost-efficient."
        else:
            winner = "model_b"
            winner_model = payload.model_b
            reason = f"Quality score tied (~{res_b['quality_score']:.2f}), but {payload.model_b} is more cost-efficient."
    elif score_diff > 0:
        winner = "model_a"
        winner_model = payload.model_a
        reason = f"{payload.model_a} achieved higher overall quality score ({res_a['quality_score']:.2f} vs {res_b['quality_score']:.2f})."
    else:
        winner = "model_b"
        winner_model = payload.model_b
        reason = f"{payload.model_b} achieved higher overall quality score ({res_b['quality_score']:.2f} vs {res_a['quality_score']:.2f})."

    return ABCompareResponse(
        model_a=payload.model_a,
        eval_a=eval_response_a,
        model_b=payload.model_b,
        eval_b=eval_response_b,
        winner=winner,
        winner_model=winner_model,
        efficiency_ratio=eff_ratio,
        recommendation_reason=reason,
    )
