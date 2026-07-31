"""
demos/routing_simple.py
=======================
Scenario: Model Routing — Simple Query
A trivial arithmetic question is sent requesting gpt-4o.
The model router detects LOW complexity and downgrades it to a cheaper model
(e.g. gemini-2.0-flash), saving cost without sacrificing quality.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "routing_simple"
SCENARIO_NAME = "Routing — Simple Query"
DESCRIPTION = (
    "Simple arithmetic requested on gpt-4o. "
    "Router detects LOW complexity and downgrades to the cheapest model."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    prompt = "What is 347 multiplied by 28?"
    step = call_gateway(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=True,
        bypass_routing=False,   # routing ON
        label="Simple Query — Routed to Cheaper Model",
    )
    result.steps.append(step)

    result.status = "error" if step.status == "error" else "success"
    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
