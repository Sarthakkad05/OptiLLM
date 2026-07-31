"""
demos/full_pipeline.py
======================
Scenario: Full Optimization Pipeline (End-to-End)
Demonstrates all 4 OptiLLM optimizations firing in a 4-step sequence:

  Step 1 — Cache Miss       : First request on a fresh topic (all features on)
  Step 2 — Compression      : Bloated prompt compressed before LLM call
  Step 3 — Routing          : Simple query downgraded to cheap model
  Step 4 — Cache Hit        : Same topic as Step 1, served from FAISS

This is the "money shot" scenario for demos — it shows every layer of the
optimization stack in a single, coherent narrative.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "full_pipeline"
SCENARIO_NAME = "Full Optimization Pipeline"
DESCRIPTION = (
    "All 4 optimizations in sequence: "
    "Cache Miss → Compression → Routing → Cache Hit."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    # ── Step 1: Cache Miss ────────────────────────────────────────────────────
    step1 = call_gateway(
        messages=[{
            "role": "user",
            "content": "What is the cancellation policy for annual subscriptions?"
        }],
        model="gpt-4o",
        bypass_cache=False,
        bypass_compression=True,
        bypass_routing=True,
        label="Step 1 — Cache Miss (cold start, full LLM call)",
    )
    result.steps.append(step1)

    # ── Step 2: Context Compression ───────────────────────────────────────────
    bloated_system = (
        "You are a helpful billing support agent.\n"
        + "Always verify the customer's account before proceeding.\n" * 15
        + "Maintain a professional tone at all times.\n" * 15
        + "Escalate complex issues to your supervisor.\n" * 10
    )
    step2 = call_gateway(
        messages=[
            {"role": "system", "content": bloated_system},
            {"role": "user", "content": "How do I update my payment method?"},
        ],
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=False,
        bypass_routing=True,
        label="Step 2 — Context Compression (bloated system prompt trimmed)",
        prompt_override=f"[Bloated system: {len(bloated_system.split())} words] How do I update my payment method?",
    )
    result.steps.append(step2)

    # ── Step 3: Model Routing ─────────────────────────────────────────────────
    step3 = call_gateway(
        messages=[{
            "role": "user",
            "content": "What is the square root of 144?"
        }],
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=True,
        bypass_routing=False,
        label="Step 3 — Model Routing (simple query downgraded)",
    )
    result.steps.append(step3)

    # ── Step 4: Cache Hit ─────────────────────────────────────────────────────
    step4 = call_gateway(
        messages=[{
            "role": "user",
            "content": "Can you explain the cancellation terms for yearly subscriptions?"
        }],
        model="gpt-4o",
        bypass_cache=False,
        bypass_compression=True,
        bypass_routing=True,
        label="Step 4 — Cache Hit (semantically matches Step 1)",
    )
    result.steps.append(step4)

    errors = [s for s in result.steps if s.status == "error"]
    if len(errors) == len(result.steps):
        result.status = "error"
    elif errors:
        result.status = "partial"
    else:
        result.status = "success"

    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
