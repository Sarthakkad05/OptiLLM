"""
demos/routing_complex.py
========================
Scenario: Model Routing — Complex Query
A multi-step algorithm implementation request is sent on gpt-4o.
The model router detects HIGH complexity and keeps it on the frontier model
— no routing, no downgrade.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "routing_complex"
SCENARIO_NAME = "Routing — Complex Query"
DESCRIPTION = (
    "Complex coding task requested on gpt-4o. "
    "Router detects HIGH complexity and keeps the frontier model — no downgrade."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    prompt = (
        "Implement a Python function that solves the 0/1 knapsack problem using "
        "dynamic programming with memoization. Include detailed comments explaining "
        "each step of the algorithm, analyze its time and space complexity, and "
        "provide example test cases with expected outputs."
    )
    step = call_gateway(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=True,
        bypass_routing=False,   # routing ON
        label="Complex Query — Frontier Model Retained",
    )
    result.steps.append(step)

    result.status = "error" if step.status == "error" else "success"
    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
