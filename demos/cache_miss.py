"""
demos/cache_miss.py
===================
Scenario: Cache Miss
Shows a first-time request where the FAISS cache is empty.
The request flows all the way to the LLM provider.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "cache_miss"
SCENARIO_NAME = "Cache Miss"
DESCRIPTION = "First-time request — cache is cold, full LLM call is made and logged."


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    prompt = "What is the 30-day return policy for online purchases?"
    step = call_gateway(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        bypass_cache=False,
        bypass_compression=True,   # isolate cache behaviour
        bypass_routing=True,       # isolate cache behaviour
        label="Cache Miss — Cold Request",
    )
    result.steps.append(step)

    if step.status == "error":
        result.status = "error"
    else:
        result.status = "success"

    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
