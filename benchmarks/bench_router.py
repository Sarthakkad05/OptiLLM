#!/usr/bin/env python3
"""
OptiLLM Router Benchmark
Measures router classification latency, tier decision accuracy, and overhead.

Usage:
    python benchmarks/bench_router.py
"""

import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.router import route

BENCHMARK_PROMPTS = [
    # Low complexity (expected to route to cheaper model like gpt-4o-mini / gemini-2.0-flash)
    ("What is the capital of Australia?", "low"),
    ("How many centimeters in 5 inches?", "low"),
    ("Define photosynthesis in simple terms.", "low"),
    ("Translate 'good morning' to Spanish.", "low"),
    ("What year was the Moon landing?", "low"),

    # Medium complexity
    ("Summarize the economic consequences of high inflation on emerging markets.", "medium"),
    ("Write a Python function to check if a binary tree is symmetric.", "medium"),
    ("Explain the difference between optimistic and pessimistic locking in databases.", "medium"),

    # High complexity (should preserve requested frontier model like gpt-4o / claude-3-5-sonnet)
    ("Design a distributed multi-datacenter consensus protocol with fault tolerance against Byzantine nodes.", "high"),
    ("Analyze the security implications of quantum Shor's algorithm on RSA-4096 vs ECDSA-256.", "high"),
    ("Write a complete compiler front-end parser in Rust with AST construction and error recovery.", "high"),
]


def run_router_benchmark() -> Dict:
    latencies_ms = []
    decisions = []

    for prompt, expected_complexity in BENCHMARK_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        start = time.perf_counter()
        result = route(messages, requested_model="gpt-4o")
        latency = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency)

        decisions.append({
            "prompt": prompt[:40],
            "expected": expected_complexity,
            "complexity": result["complexity"],
            "routed": result["routed"],
            "model_used": result["model_used"],
            "latency_ms": latency,
        })

    sorted_lats = sorted(latencies_ms)
    p50 = sorted_lats[len(sorted_lats) // 2]
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)]

    routed_count = sum(1 for d in decisions if d["routed"])

    return {
        "total_samples": len(BENCHMARK_PROMPTS),
        "routed_pct": (routed_count / len(BENCHMARK_PROMPTS)) * 100,
        "avg_latency_ms": statistics.mean(latencies_ms),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "decisions": decisions,
    }


def main():
    print("🔬 Running OptiLLM Intelligent Router Benchmark...")
    results = run_router_benchmark()

    print(f"\nRouter Overhead & Latency:")
    print(f"  • P50 Latency : {results['p50_latency_ms']:.2f}ms")
    print(f"  • P95 Latency : {results['p95_latency_ms']:.2f}ms")
    print(f"  • Avg Latency : {results['avg_latency_ms']:.2f}ms")
    print(f"  • Down-routed : {results['routed_pct']:.1f}%\n")

    print("Sample Decision Summary:")
    print("  Prompt Snippet                           | Expected | Assessed | Routed | Target Model")
    print("  ─────────────────────────────────────────┼──────────┼──────────┼────────┼─────────────")
    for d in results["decisions"]:
        r_str = "Yes" if d["routed"] else "No"
        print(f"  {d['prompt'].ljust(40)} | {d['expected'].ljust(8)} | {d['complexity'].ljust(8)} | {r_str.ljust(6)} | {d['model_used']}")
    print()


if __name__ == "__main__":
    main()
