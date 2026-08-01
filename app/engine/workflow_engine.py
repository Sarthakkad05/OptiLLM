"""
LangGraph Workflow Engine.
Replaces linear request processing with a graph-based state machine supporting
conditional quality retries, context compression, cache lookups, and model routing.
"""

import logging
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.engine.cache import check_cache, insert_cache
from app.engine.compressor import compress
from app.engine.router import route
from app.providers.dispatcher import call_provider

logger = logging.getLogger("optillm.engine.workflow_engine")


class GatewayState(TypedDict):
    messages: List[Dict[str, Any]]
    model_requested: str
    model_used: str
    messages_to_send: List[Dict[str, Any]]
    cache_hit: bool
    compressed: bool
    routed: bool
    tokens_input: int
    tokens_output: int
    tokens_saved: int
    cost_usd: float
    savings_usd: float
    latency_ms: int
    content: str
    quality_score: float
    retry_count: int
    db: Any
    routing_reason: Optional[str]


def check_cache_node(state: GatewayState) -> Dict[str, Any]:
    """Node: Check distributed Redis / FAISS semantic cache."""
    db = state.get("db")
    if not db:
        return {}

    cached = check_cache(state["messages"], db)
    if cached:
        return {
            "cache_hit": True,
            "content": cached["response_text"],
            "model_used": cached["model"],
            "tokens_input": cached["tokens_input"],
            "tokens_output": cached["tokens_output"],
            "cost_usd": 0.0,
            "quality_score": 1.0,
        }

    return {"cache_hit": False}


def compress_context_node(state: GatewayState) -> Dict[str, Any]:
    """Node: Apply heuristic cleaning & context compression."""
    if state.get("cache_hit"):
        return {}

    compressed_msgs, stats = compress(
        messages=state["messages"], model=state["model_requested"]
    )
    return {
        "messages_to_send": compressed_msgs,
        "compressed": stats["was_compressed"],
        "tokens_saved": stats["tokens_saved"],
    }


def classify_and_route_node(state: GatewayState) -> Dict[str, Any]:
    """Node: Classify prompt complexity and select optimal model."""
    if state.get("cache_hit"):
        return {}

    messages = state.get("messages_to_send") or state["messages"]
    model_req = state["model_requested"]

    # If retrying due to low quality score, upgrade to high tier model
    if state.get("retry_count", 0) > 0:
        return {
            "model_used": "gpt-4o",
            "routed": True,
            "routing_reason": "Quality retry upgrade to gpt-4o",
        }

    decision = route(messages=messages, requested_model=model_req)
    return {
        "model_used": decision.get("model_used", model_req),
        "routed": decision.get("routed", False),
        "routing_reason": decision.get("routing_reason"),
    }


async def call_provider_node(state: GatewayState) -> Dict[str, Any]:
    """Node: Async provider invocation."""
    if state.get("cache_hit"):
        return {}

    start_time = time.perf_counter()
    messages = state.get("messages_to_send") or state["messages"]
    model = state["model_used"]

    response = await call_provider(
        messages=messages,
        model=model,
        temperature=0.7,
    )

    latency = int((time.perf_counter() - start_time) * 1000)

    content = response.get("content", "")
    tokens_in = response.get("tokens_input", 0)
    tokens_out = response.get("tokens_output", 0)

    return {
        "content": content,
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "latency_ms": latency,
        "cost_usd": 0.001,  # Estimated execution cost
    }


def evaluate_quality_node(state: GatewayState) -> Dict[str, Any]:
    """Node: Evaluates completion response quality (0.0 to 1.0)."""
    if state.get("cache_hit"):
        return {"quality_score": 1.0}

    content = state.get("content", "")
    # Simple quality metric: non-empty & sufficient detail
    score = 0.95 if len(content.strip()) > 10 else 0.4
    return {"quality_score": score}


def should_retry_or_finish(state: GatewayState) -> str:
    """Conditional Edge: Retry with higher model if quality score is low (< 0.7)."""
    if state.get("cache_hit"):
        return "finish"

    score = state.get("quality_score", 1.0)
    retries = state.get("retry_count", 0)

    if score < 0.7 and retries < 2:
        logger.warning("Quality score %.2f < 0.7 — triggering retry node.", score)
        return "retry"
    return "finish"


def insert_cache_node(state: GatewayState) -> Dict[str, Any]:
    """Node: Cache entry insertion."""
    if state.get("cache_hit") or not state.get("db"):
        return {}

    insert_cache(
        messages=state["messages"],
        response_text=state["content"],
        model=state["model_used"],
        tokens_input=state["tokens_input"],
        tokens_output=state["tokens_output"],
        db=state["db"],
    )
    return {}


# ── LangGraph Workflow Assembly ───────────────────────────────────────────────


def create_gateway_workflow() -> StateGraph:
    """Compiles the OptiLLM state graph workflow."""
    workflow = StateGraph(GatewayState)

    # Add Nodes
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("compress_context", compress_context_node)
    workflow.add_node("classify_and_route", classify_and_route_node)
    workflow.add_node("call_provider", call_provider_node)
    workflow.add_node("evaluate_quality", evaluate_quality_node)
    workflow.add_node("insert_cache", insert_cache_node)

    # Add Edges
    workflow.add_edge(START, "check_cache")
    workflow.add_edge("check_cache", "compress_context")
    workflow.add_edge("compress_context", "classify_and_route")
    workflow.add_edge("classify_and_route", "call_provider")
    workflow.add_edge("call_provider", "evaluate_quality")

    # Conditional quality retry loop
    workflow.add_conditional_edges(
        "evaluate_quality",
        should_retry_or_finish,
        {
            "retry": "classify_and_route",
            "finish": "insert_cache",
        },
    )
    workflow.add_edge("insert_cache", END)

    return workflow.compile()


compiled_gateway_graph = create_gateway_workflow()


async def run_workflow_pipeline(
    messages: List[Dict[str, Any]],
    model: str,
    db: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Async entrypoint running a completion request through the LangGraph workflow engine.
    """
    initial_state: GatewayState = {
        "messages": messages,
        "model_requested": model,
        "model_used": model,
        "messages_to_send": messages,
        "cache_hit": False,
        "compressed": False,
        "routed": False,
        "tokens_input": 0,
        "tokens_output": 0,
        "tokens_saved": 0,
        "cost_usd": 0.0,
        "savings_usd": 0.0,
        "latency_ms": 0,
        "content": "",
        "quality_score": 0.0,
        "retry_count": 0,
        "db": db,
        "routing_reason": None,
    }

    final_state = await compiled_gateway_graph.ainvoke(initial_state)
    return final_state
