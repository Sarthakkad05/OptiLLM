#!/usr/bin/env python3
"""
OptiLLM Live-Traffic Benchmark
Validates router down-routing, context compression, and cache-vs-direct latency
against REAL OpenAI API calls and REAL LLM-as-judge quality scoring.

This is the live counterpart to bench_router.py / bench_compression.py / bench_cache.py,
which only measure the pipeline's own processing overhead using synthetic data. This
script answers the question those can't: when the router or compressor changes what's
sent to the model, does response quality actually hold up?

Scope: OpenAI only (gpt-4o / gpt-4o-mini), for now — see docs/benchmarks.md for why.
Cost control: small fixed prompt sets, cheap judge model, calls both models only when
a real trade-off is being tested. Actual spend is measured and reported at the end
using published OpenAI list prices.

Requires OPENAI_API_KEY set in .env (real key, not a placeholder).

Usage:
    python benchmarks/bench_live.py
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.providers.dispatcher as dispatcher
from app.core.config import settings
from app.engine.compressor import compress
from app.engine.router import route
from app.evaluation.judge import get_judge

# Approximate OpenAI list prices, USD per 1M tokens (input, output) — for cost reporting only.
_PRICES = {
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
}

ROUTER_PROMPTS = [
    ("What is the capital of Australia?", "low"),
    ("How many centimeters in 5 inches?", "low"),
    ("Define photosynthesis in simple terms.", "low"),
    ("Summarize the economic consequences of high inflation on emerging markets.", "medium"),
    ("Write a Python function to check if a binary tree is symmetric.", "medium"),
    ("Design a distributed multi-datacenter consensus protocol with fault tolerance against Byzantine nodes.", "high"),
]

COMPRESSION_CONVERSATIONS = [
    ("medium", [
        {"role": "system", "content": "You are a database consultant."},
        {"role": "user", "content": "Here is our schema:\n\nCREATE TABLE users (id INT, name TEXT);\n---\n---\nCREATE TABLE orders (id INT, user_id INT);\n\n\n\nHow do we index this?"},
        {"role": "assistant", "content": "Create a foreign key index on orders(user_id)."},
        {"role": "user", "content": "How do we index this?\n\nCREATE TABLE orders (id INT, user_id INT);"},
    ]),
    ("large", [
        {"role": "system", "content": "You are an enterprise AI architecture specialist."},
        {"role": "user", "content": "In our current infrastructure, we have legacy microservices running across three AWS regions with high latency. " * 15},
        {"role": "assistant", "content": "I recommend deploying an API gateway layer with regional edge caching and DynamoDB global tables."},
        {"role": "user", "content": "We also have intermittent network timeouts between our US-East and EU-West clusters during peak hours. " * 20},
        {"role": "assistant", "content": "You should configure mutual TLS with connection pooling and circuit breakers."},
        {"role": "user", "content": "What specific timeout thresholds and backoff retry algorithms should we configure for the inter-region RPCs?"},
    ]),
]

LATENCY_PROMPT = "What is the speed of light in vacuum?"
LATENCY_SAMPLES = 5

_spend_tracker: List[Dict[str, Any]] = []


def _price_for(model: str) -> tuple:
    # OpenAI returns a versioned model string (e.g. "gpt-4o-mini-2024-07-18"); match the
    # longest family prefix first so "gpt-4o-mini-..." doesn't get mis-matched to "gpt-4o".
    for family in sorted(_PRICES, key=len, reverse=True):
        if model.startswith(family):
            return _PRICES[family]
    return (0.0, 0.0)


def _track_spend(model: str, tokens_input: int, tokens_output: int) -> float:
    price_in, price_out = _price_for(model)
    cost = (tokens_input / 1_000_000) * price_in + (tokens_output / 1_000_000) * price_out
    _spend_tracker.append({"model": model, "tokens_input": tokens_input, "tokens_output": tokens_output, "cost_usd": cost})
    return cost


# Patch dispatcher.call_provider so EVERY real call — including the ones the LLM judge
# makes internally via `from app.providers.dispatcher import call_provider` — gets its
# spend tracked, not just the ones this script calls directly.
_original_call_provider = dispatcher.call_provider


async def _tracked_call_provider(messages, model, temperature: float = 0.7, max_tokens=None):
    result = await _original_call_provider(messages, model, temperature, max_tokens)
    _track_spend(result.get("model", model), result.get("tokens_input", 0), result.get("tokens_output", 0))
    return result


dispatcher.call_provider = _tracked_call_provider


async def _call(messages: List[Dict], model: str, max_tokens: int = 300) -> Dict[str, Any]:
    return await dispatcher.call_provider(messages, model=model, temperature=0.3, max_tokens=max_tokens)


async def bench_router_quality() -> Dict[str, Any]:
    """For each prompt, compare judged quality of the requested model vs. the router's
    chosen (often cheaper) model — the thing the local-only router benchmark can't tell you."""
    judge = get_judge()
    rows = []

    for prompt, expected in ROUTER_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        decision = route(messages, requested_model="gpt-4o")
        routed_model = decision["model_used"]

        baseline = await _call(messages, "gpt-4o")
        baseline_eval = await judge.evaluate_with_llm(messages, baseline["content"])

        if decision["routed"] and routed_model != "gpt-4o":
            routed = await _call(messages, routed_model)
            routed_eval = await judge.evaluate_with_llm(messages, routed["content"])
        else:
            routed = baseline
            routed_eval = baseline_eval

        rows.append({
            "prompt": prompt[:50],
            "expected_complexity": expected,
            "assessed_complexity": decision["complexity"].value,
            "routed": decision["routed"],
            "routed_model": routed_model,
            "baseline_quality": baseline_eval["quality_score"],
            "routed_quality": routed_eval["quality_score"],
            "quality_delta": round(routed_eval["quality_score"] - baseline_eval["quality_score"], 4),
        })

    return {"rows": rows}


async def bench_compression_quality() -> Dict[str, Any]:
    """For each conversation, compare judged quality of an answer generated from the
    smart-compressed context vs. the minimally-cleaned (uncompressed) context, judged
    against the original full conversation — does compression cost you real quality?"""
    judge = get_judge()
    rows = []

    for label, convo in COMPRESSION_CONVERSATIONS:
        uncompressed_msgs, uncompressed_stats = compress(convo, model="gpt-4o-mini", max_tokens=300, mode="minimal")
        compressed_msgs, compressed_stats = compress(convo, model="gpt-4o-mini", max_tokens=300, mode="smart")

        uncompressed_resp = await _call(uncompressed_msgs, "gpt-4o-mini")
        compressed_resp = await _call(compressed_msgs, "gpt-4o-mini")

        # Judge both against the ORIGINAL full conversation, since that's what the user actually asked.
        uncompressed_eval = await judge.evaluate_with_llm(convo, uncompressed_resp["content"])
        compressed_eval = await judge.evaluate_with_llm(convo, compressed_resp["content"])

        token_reduction_pct = round(
            (1 - compressed_stats["compressed_tokens"] / max(1, uncompressed_stats["compressed_tokens"])) * 100, 1
        )
        rows.append({
            "conversation": label,
            "token_reduction_pct": token_reduction_pct,
            "uncompressed_quality": uncompressed_eval["quality_score"],
            "compressed_quality": compressed_eval["quality_score"],
            "quality_delta": round(compressed_eval["quality_score"] - uncompressed_eval["quality_score"], 4),
        })

    return {"rows": rows}


async def bench_direct_call_latency() -> Dict[str, Any]:
    """Real end-to-end latency of an uncached OpenAI call, to compare against the
    already-measured local cache-hit latency (~8ms) — the number the synthetic
    benchmark could only estimate as a 'reference figure'."""
    latencies_ms = []
    messages = [{"role": "user", "content": LATENCY_PROMPT}]
    for _ in range(LATENCY_SAMPLES):
        start = time.perf_counter()
        await _call(messages, "gpt-4o-mini", max_tokens=100)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    sorted_lats = sorted(latencies_ms)
    return {
        "samples": latencies_ms,
        "avg_ms": statistics.mean(latencies_ms),
        "p50_ms": sorted_lats[len(sorted_lats) // 2],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def _check_prerequisites():
    placeholder_values = {"", "your_openai_api_key_here"}
    if settings.OPENAI_API_KEY in placeholder_values:
        print("ERROR: OPENAI_API_KEY is not set (or is a placeholder) in .env.")
        print("This benchmark makes real, billed OpenAI API calls and requires a real key.")
        sys.exit(1)


async def main():
    _check_prerequisites()

    print("Running live router-quality benchmark (real OpenAI calls + LLM judge)...")
    router_results = await bench_router_quality()

    print("Running live compression-quality benchmark (real OpenAI calls + LLM judge)...")
    compression_results = await bench_compression_quality()

    print(f"Running direct-call latency benchmark ({LATENCY_SAMPLES} real OpenAI calls)...")
    latency_results = await bench_direct_call_latency()

    total_cost = sum(r["cost_usd"] for r in _spend_tracker)
    total_calls = len(_spend_tracker)

    print("\n" + "=" * 70)
    print("LIVE BENCHMARK — Router Quality (baseline gpt-4o vs. routed model)")
    print("=" * 70)
    for row in router_results["rows"]:
        print(f"  [{row['assessed_complexity']:>6}] routed={row['routed']!s:5} -> {row['routed_model']:15} "
              f"| baseline_q={row['baseline_quality']:.2f} routed_q={row['routed_quality']:.2f} "
              f"delta={row['quality_delta']:+.2f} | {row['prompt']}")

    print("\n" + "=" * 70)
    print("LIVE BENCHMARK — Compression Quality (smart vs. minimal, judged vs. original)")
    print("=" * 70)
    for row in compression_results["rows"]:
        print(f"  [{row['conversation']:>6}] uncompressed_q={row['uncompressed_quality']:.2f} "
              f"compressed_q={row['compressed_quality']:.2f} delta={row['quality_delta']:+.2f}")

    print("\n" + "=" * 70)
    print("LIVE BENCHMARK — Direct (uncached) Call Latency")
    print("=" * 70)
    print(f"  avg={latency_results['avg_ms']:.0f}ms  p50={latency_results['p50_ms']:.0f}ms  "
          f"min={latency_results['min_ms']:.0f}ms  max={latency_results['max_ms']:.0f}ms")

    print("\n" + "=" * 70)
    print(f"Total real API calls: {total_calls}  |  Measured spend: ${total_cost:.4f}")
    print("=" * 70)

    _write_report(router_results, compression_results, latency_results, total_cost, total_calls)


def _write_report(router_results, compression_results, latency_results, total_cost, total_calls):
    from datetime import datetime, timezone

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# OptiLLM Live-Traffic Benchmark Report",
        "",
        f"> **Generated:** {now_str}  ",
        "> **Environment:** Real OpenAI API calls (gpt-4o, gpt-4o-mini), real LLM-as-judge scoring.  ",
        f"> **Measured spend for this run:** ${total_cost:.4f} across {total_calls} API calls.",
        "",
        "This report exists to answer what `benchmarks/BENCHMARK_REPORT.md` (local/synthetic) cannot: "
        "when OptiLLM routes to a cheaper model or compresses context, does judged answer quality hold up "
        "against real provider responses? Scope is OpenAI-only for this pass.",
        "",
        "---",
        "",
        "## 1. Router Quality: Requested Model vs. Routed Model",
        "",
        "Quality scored 0.0-1.0 by GPT-4o-mini acting as an LLM judge (correctness, relevance, "
        "completeness, conciseness, safety — see `app/evaluation/judge.py`). A negative delta means "
        "the routed (cheaper) model scored lower than the originally requested model on the same prompt.",
        "",
        "| Complexity | Routed? | Routed Model | Baseline Quality (gpt-4o) | Routed Quality | Delta | Prompt |",
        "|---|:---:|---|:---:|:---:|:---:|---|",
    ]
    for row in router_results["rows"]:
        lines.append(
            f"| {row['assessed_complexity']} | {'Yes' if row['routed'] else 'No'} | `{row['routed_model']}` | "
            f"{row['baseline_quality']:.2f} | {row['routed_quality']:.2f} | {row['quality_delta']:+.2f} | {row['prompt']} |"
        )

    deltas = [r["quality_delta"] for r in router_results["rows"] if r["routed"]]
    if deltas:
        lines += [
            "",
            f"**Average quality delta on routed prompts:** {statistics.mean(deltas):+.4f} "
            f"({'quality held up' if statistics.mean(deltas) >= -0.05 else 'measurable quality loss on down-routing'}).",
        ]

    misclassified = [
        r for r in router_results["rows"] if r["expected_complexity"] != r["assessed_complexity"]
    ]
    if misclassified:
        lines += ["", "**Rule-based router misclassifications observed this run:**", ""]
        for r in misclassified:
            lines.append(
                f"- Expected `{r['expected_complexity']}`, rule-based router assessed `{r['assessed_complexity']}` "
                f"for: \"{r['prompt']}\" — see per-run console log for whether the shadow-mode AI router "
                "disagreed (it logs `Shadow mode disagreement: ...` when its own classification differs)."
            )

    lines += [
        "",
        "---",
        "",
        "## 2. Compression Quality: Smart-Compressed vs. Minimal Context",
        "",
        "Both variants are judged against the **original, uncompressed conversation** — the question is "
        "whether the smart-compressed prompt still lets the model answer what the user actually asked.",
        "",
        "| Conversation | Token Reduction | Uncompressed Quality | Compressed Quality | Delta |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for row in compression_results["rows"]:
        lines.append(
            f"| {row['conversation']} | {row['token_reduction_pct']:.1f}% | {row['uncompressed_quality']:.2f} | "
            f"{row['compressed_quality']:.2f} | {row['quality_delta']:+.2f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Direct (Uncached) Call Latency — Real Provider Round Trip",
        "",
        f"Measured over {LATENCY_SAMPLES} real `gpt-4o-mini` calls with a short prompt, "
        "for comparison against the local semantic-cache hit latency (~8ms, see `benchmarks/BENCHMARK_REPORT.md`):",
        "",
        f"- **Average:** {latency_results['avg_ms']:.0f} ms",
        f"- **P50:** {latency_results['p50_ms']:.0f} ms",
        f"- **Min / Max:** {latency_results['min_ms']:.0f} ms / {latency_results['max_ms']:.0f} ms",
        "",
        "---",
        "",
        "## Known Limitations of This Run",
        "",
        "- OpenAI only — Anthropic/Gemini/Groq/Mistral not yet validated live.",
        "- Small sample sizes (a handful of prompts per category), chosen to bound cost — not statistically rigorous.",
        "- The judge model (gpt-4o-mini) grading responses from gpt-4o-mini in the compression test introduces "
        "some same-model bias; a stronger/independent judge model would be a good follow-up.",
        "- No concurrency — this measures single-request behavior, not production load.",
        "",
    ]

    report_path = os.path.join(os.path.dirname(__file__), "LIVE_BENCHMARK_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nLive benchmark report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
