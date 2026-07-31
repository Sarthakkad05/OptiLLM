"""
demos/cache_hit.py
==================
Scenario: Cache Hit
Sends a semantically similar (but differently worded) query that should
match the entry seeded by the Cache Miss scenario. Demonstrates instant
sub-20ms response with zero token cost.

NOTE: Run "Cache Miss" first so the cache has an entry for the return-policy
topic. If the cache is cold this will show a miss instead.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "cache_hit"
SCENARIO_NAME = "Cache Hit"
DESCRIPTION = (
    "Semantically similar query served instantly from FAISS cache — "
    "0 tokens billed, ~10ms latency."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    # Deliberately different wording from cache_miss.py — but same semantic intent
    prompt = "How can I return an item I bought online within a month?"
    step = call_gateway(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        bypass_cache=False,
        bypass_compression=True,
        bypass_routing=True,
        label="Cache Hit — Semantically Similar Query",
    )
    result.steps.append(step)

    if step.status == "error":
        result.status = "error"
    elif not step.cache_hit:
        # Still succeeded but no hit — flag as partial for the UI
        result.status = "partial"
    else:
        result.status = "success"

    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
