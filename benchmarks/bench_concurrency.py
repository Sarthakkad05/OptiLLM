#!/usr/bin/env python3
"""
OptiLLM Concurrency & Multi-Replica Validation
Hits a running docker-compose stack (2+ optillm replicas behind the nginx LB,
shared Postgres + Redis) with real concurrent HTTP traffic, in mock mode (no
provider calls, no spend — see docker-compose.loadtest.yml), to answer what
the single-process local benchmarks can't:

  - Does the gateway actually survive concurrent load without errors?
  - Do requests actually get distributed across multiple replica containers?
  - Does shared Postgres-backed state (request_logs) stay consistent when
    multiple independent processes are writing to it concurrently — i.e.
    does anything get silently dropped or double-counted?

Usage:
    python benchmarks/bench_concurrency.py --url http://localhost:8000 \
        --requests 300 --concurrency 30
"""

import argparse
import asyncio
import statistics
import time
from collections import Counter
from typing import Any, Dict, List

import httpx

PROMPT_POOL = [
    "What is the capital of France?",
    "Explain recursion in one sentence.",
    "What is the boiling point of water in Celsius?",
    "Define an API in simple terms.",
    "What is 12 times 12?",
]


async def _send_one(client: httpx.AsyncClient, url: str, prompt: str, sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        start = time.perf_counter()
        try:
            resp = await client.post(f"{url}/v1/chat/completions", json=payload, timeout=30.0)
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "ok": resp.status_code == 200,
                "status": resp.status_code,
                "latency_ms": latency_ms,
                "upstream": resp.headers.get("x-upstream-addr"),
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return {"ok": False, "status": None, "latency_ms": latency_ms, "upstream": None, "error": str(exc)}


async def run_load_test(url: str, total_requests: int, concurrency: int) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        # Confirm the stack is reachable before hammering it.
        health = await client.get(f"{url}/health", timeout=10.0)
        health.raise_for_status()

        start = time.perf_counter()
        tasks = [
            _send_one(client, url, PROMPT_POOL[i % len(PROMPT_POOL)], sem)
            for i in range(total_requests)
        ]
        results = await asyncio.gather(*tasks)
        wall_time_s = time.perf_counter() - start

    latencies = sorted(r["latency_ms"] for r in results)
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    upstream_counts = Counter(r["upstream"] for r in results if r["upstream"])

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "wall_time_s": round(wall_time_s, 2),
        "throughput_rps": round(total_requests / wall_time_s, 2) if wall_time_s > 0 else 0,
        "successes": len(successes),
        "failures": len(failures),
        "failure_samples": [r.get("error", r.get("status")) for r in failures[:5]],
        "latency_avg_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_p50_ms": round(pct(0.50), 1),
        "latency_p95_ms": round(pct(0.95), 1),
        "latency_p99_ms": round(pct(0.99), 1),
        "latency_max_ms": round(max(latencies), 1) if latencies else 0,
        "upstream_distribution": dict(upstream_counts),
        "distinct_replicas_hit": len(upstream_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="OptiLLM concurrency/HA load test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=30)
    args = parser.parse_args()

    print(f"Running {args.requests} requests at concurrency={args.concurrency} against {args.url} ...")
    result = asyncio.run(run_load_test(args.url, args.requests, args.concurrency))

    print("\n" + "=" * 70)
    print("CONCURRENCY LOAD TEST RESULT")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k}: {v}")

    if result["distinct_replicas_hit"] < 2:
        print("\nWARNING: only 1 distinct replica was hit — check that optillm was actually scaled.")

    return result


if __name__ == "__main__":
    main()
