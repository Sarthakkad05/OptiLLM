"""
demos/compression_large.py
==========================
Scenario: Context Compression — Severe Bloat
A massively bloated system prompt with 80+ repeated sentences (~1400 words).
The compressor should save hundreds of tokens. Demonstrates the extreme case
developers accidentally ship in production.
"""

import time
from demos.base import DemoResult, call_gateway

SCENARIO_KEY = "compression_large"
SCENARIO_NAME = "Compression — Severe Bloat"
DESCRIPTION = (
    "80+ repetitions in system prompt (~1,400 words). "
    "Compressor aggressively trims, saving hundreds of tokens."
)


def run() -> DemoResult:
    result = DemoResult(
        scenario_key=SCENARIO_KEY,
        scenario_name=SCENARIO_NAME,
        description=DESCRIPTION,
    )
    t0 = time.time()

    # Simulate a real-world bloated production system prompt
    bloated_system = (
        "You are a professional customer service assistant.\n\n"
        "Company Policy:\n"
        + "Our company values customer satisfaction above all else. "
          "We strive to provide the best possible service at all times.\n" * 40
        + "\nPlease always be polite and professional.\n" * 20
        + "\nRemember to follow all guidelines at all times.\n" * 20
        + "\nAll customer data must be kept confidential.\n" * 10
    )

    word_count = len(bloated_system.split())
    messages = [
        {"role": "system", "content": bloated_system},
        {"role": "user", "content": "What is your company's refund policy?"},
    ]

    step = call_gateway(
        messages=messages,
        model="gpt-4o",
        bypass_cache=True,
        bypass_compression=False,
        bypass_routing=True,
        label=f"Compression — Severe Bloat ({word_count:,} words in system prompt)",
        prompt_override=f"[System: {word_count:,} words bloated prompt] + refund policy question",
    )
    result.steps.append(step)

    result.status = "error" if step.status == "error" else "success"
    result.elapsed_ms = int((time.time() - t0) * 1000)
    result.compute_totals()
    return result
