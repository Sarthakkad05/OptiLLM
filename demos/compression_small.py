"""
demos/compression_small.py
==========================
Scenario: Context Compression — Moderate Bloat
A customer-service system prompt with ~30 repeated sentences is compressed
before being forwarded to the LLM. Shows modest token reduction.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "compression_small"
SCENARIO_NAME = "Compression — Moderate Bloat"
DESCRIPTION = "~30 repeated sentences in system prompt. Compressor trims redundant content before calling LLM."


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    bloated_system = (
        "You are a helpful customer service assistant for Acme Corp.\n\n"
        "Always greet the customer warmly.\n" * 10
        + "Be polite and professional at all times.\n" * 10
        + "Follow all company guidelines when responding.\n" * 10
        + "\nOur company values customer satisfaction above all else."
    )

    messages = [
        {"role": "system", "content": bloated_system},
        {"role": "user", "content": "What are your customer service hours?"},
    ]

    step = call_gateway(
        messages=messages,
        model="gpt-4o",
        bypass_cache=True,        # isolate compression
        bypass_compression=False,
        bypass_routing=True,
        label="Compression — Moderate Bloat",
        prompt_override=f"[System: {len(bloated_system.split())} words] + user question",
    )
    result.steps.append(step)

    result.status = "error" if step.status == "error" else "success"
    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
