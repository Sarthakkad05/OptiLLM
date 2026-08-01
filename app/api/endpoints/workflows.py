"""
POST /api/v1/workflows/run
POST /api/v1/workflows/compare
GET /api/v1/workflows/graph
LangGraph workflow engine endpoints for graph execution and inspection.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.engine.parallel_eval import run_parallel_model_comparison
from app.engine.workflow_engine import run_workflow_pipeline

router = APIRouter()


class WorkflowRunRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[Dict[str, Any]]


class WorkflowCompareRequest(BaseModel):
    model_a: str = "gpt-4o"
    model_b: str = "gpt-4o-mini"
    messages: List[Dict[str, Any]]


@router.post("/run", tags=["Workflows"])
async def run_workflow_endpoint(
    request: WorkflowRunRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Executes request through the LangGraph state machine workflow engine.
    """
    final_state = await run_workflow_pipeline(
        messages=request.messages,
        model=request.model,
        db=db,
    )
    # Sanitize db session from response state
    final_state.pop("db", None)
    return final_state


@router.post("/compare", tags=["Workflows"])
async def compare_models_endpoint(
    request: WorkflowCompareRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Runs dual candidate models concurrently and returns side-by-side performance metrics.
    """
    result = await run_parallel_model_comparison(
        messages=request.messages,
        model_a=request.model_a,
        model_b=request.model_b,
    )
    return result


@router.get("/graph", tags=["Workflows"])
def inspect_workflow_graph() -> Dict[str, Any]:
    """
    Returns LangGraph workflow state machine nodes, edges, and retry conditions.
    """
    return {
        "graph_name": "OptiLLM Gateway StateGraph",
        "nodes": [
            "check_cache",
            "compress_context",
            "classify_and_route",
            "call_provider",
            "evaluate_quality",
            "insert_cache",
        ],
        "conditional_edges": [
            {
                "from_node": "evaluate_quality",
                "condition": "quality_score < 0.7 and retry_count < 2",
                "if_true": "classify_and_route (Quality Retry Upgrade)",
                "if_false": "insert_cache",
            }
        ],
        "execution_mode": "async_state_machine",
    }
