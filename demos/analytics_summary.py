"""
demos/analytics_summary.py
==========================
Scenario: Analytics Snapshot
Fetches the current KPI data from /api/v1/analytics and returns it as a
DemoResult with no LLM calls. Useful at the end of a demo session to show
the cumulative financial impact of all the scenarios run.
"""

import time
from demos.base import DemoResult, DemoStep, fetch_analytics

SCENARIO_KEY = "analytics_summary"
SCENARIO_NAME = "Analytics Snapshot"
DESCRIPTION = (
    "Fetches current OptiLLM KPIs from the analytics API — "
    "total savings, cache hit rate, token economics."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    data = fetch_analytics()

    if data is None:
        result.status = "error"
        step = DemoStep(
            label="Analytics API — Failed",
            prompt="GET /api/v1/analytics",
            status="error",
            error="Could not reach analytics endpoint. Is the backend running?",
        )
        result.steps.append(step)
    else:
        summary = data.get("summary", {})
        step = DemoStep(
            label="Analytics API — Snapshot",
            prompt="GET /api/v1/analytics",
            status="success",
            cost_usd=summary.get("total_cost_usd", 0.0),
            savings_usd=summary.get("total_savings_usd", 0.0),
            tokens_saved=summary.get("total_tokens_saved", 0),
            response_preview=(
                f"{summary.get('total_requests', 0):,} requests | "
                f"{summary.get('cache_hit_rate', 0)*100:.1f}% cache hit rate | "
                f"${summary.get('total_savings_usd', 0):.6f} total saved"
            ),
        )
        result.steps.append(step)
        result.extra = data          # Full payload available to the UI
        result.status = "success"

    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
