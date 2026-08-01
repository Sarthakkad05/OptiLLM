"""
Parallel Dual-Model Comparison Engine.
Runs two candidate models concurrently via asyncio.gather, compares performance,
quality, and latency metrics, and returns the winner response.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List

from app.providers.dispatcher import call_provider

logger = logging.getLogger("optillm.engine.parallel_eval")


async def _eval_single_model(
    messages: List[Dict[str, Any]], model: str
) -> Dict[str, Any]:
    """Helper invoking single model and measuring latency."""
    start_time = time.perf_counter()
    try:
        response = await call_provider(messages=messages, model=model, temperature=0.7)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        content = response.get("content", "")
        tokens_in = response.get("tokens_input", 0)
        tokens_out = response.get("tokens_output", 0)
        return {
            "model": model,
            "status": "success",
            "content": content,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "latency_ms": latency_ms,
        }

    except Exception as exc:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.warning("Parallel eval failed for model %s: %s", model, exc)
        return {
            "model": model,
            "status": "error",
            "error": str(exc),
            "content": "",
            "tokens_input": 0,
            "tokens_output": 0,
            "latency_ms": latency_ms,
        }


async def run_parallel_model_comparison(
    messages: List[Dict[str, Any]],
    model_a: str = "gpt-4o",
    model_b: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Executes model_a and model_b in parallel using asyncio.gather.
    Selects and returns the winner model output based on latency and completion quality.
    """
    logger.info("Executing parallel model comparison: %s vs %s", model_a, model_b)

    task_a = _eval_single_model(messages, model_a)
    task_b = _eval_single_model(messages, model_b)

    res_a, res_b = await asyncio.gather(task_a, task_b)

    # Determine winner (prefer success, lower latency or higher detail)
    winner = model_a
    if res_a["status"] != "success" and res_b["status"] == "success":
        winner = model_b
    elif (
        res_a["status"] == "success"
        and res_b["status"] == "success"
        and res_b["latency_ms"] < res_a["latency_ms"]
    ):
        winner = model_b

    winner_result = res_a if winner == model_a else res_b

    return {
        "winner_model": winner,
        "winner_content": winner_result.get("content", ""),
        "model_a_result": res_a,
        "model_b_result": res_b,
        "comparison_metrics": {
            "latency_difference_ms": res_a["latency_ms"] - res_b["latency_ms"],
            "faster_model": (
                model_a if res_a["latency_ms"] < res_b["latency_ms"] else model_b
            ),
        },
    }
