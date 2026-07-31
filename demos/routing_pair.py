"""
demos/routing_pair.py
=====================
Scenario: Routing — Simple vs Complex (Side-by-Side)
Runs a simple and complex query back-to-back on gpt-4o with routing enabled.
Proves that the router correctly downgrades the easy one and keeps the hard one —
making both routing decisions visible in a single result card.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "routing_pair"
SCENARIO_NAME = "Routing — Simple vs Complex"
DESCRIPTION = (
    "Runs a simple and complex query back-to-back. "
    "Simple → downgraded. Complex → kept on frontier. Both decisions shown."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    # Step 1 — simple
    simple_prompt = "Convert 98.6 degrees Fahrenheit to Celsius."
    step1 = call_gateway(
        messages=[{"role": "user", "content": simple_prompt}],
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=True,
        bypass_routing=False,
        label="Step 1 — Simple Query (expect downgrade)",
    )
    result.steps.append(step1)

    # Step 2 — complex
    complex_prompt = (
        "Design and implement a distributed rate limiter using Redis and Python "
        "that supports sliding window counters, handles clock skew between nodes, "
        "and provides atomic operations. Explain the trade-offs versus token bucket."
    )
    step2 = call_gateway(
        messages=[{"role": "user", "content": complex_prompt}],
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=True,
        bypass_routing=False,
        label="Step 2 — Complex Query (expect frontier model kept)",
    )
    result.steps.append(step2)

    errors = [s for s in result.steps if s.status == "error"]
    if len(errors) == 2:
        result.status = "error"
    elif errors:
        result.status = "partial"
    else:
        result.status = "success"

    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
