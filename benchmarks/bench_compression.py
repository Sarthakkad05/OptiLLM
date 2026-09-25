#!/usr/bin/env python3
"""
OptiLLM Context Compression Benchmark
Measures token reduction percentages, execution latency, and compression ratios
across varying context lengths and compression modes.

Usage:
    python benchmarks/bench_compression.py
"""

import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine.compressor import compress

SAMPLE_CONVERSATIONS = [
    # Small conversation (below threshold, test pass-through & cleaning)
    [
        {"role": "system", "content": "You are a helpful software engineering assistant."},
        {"role": "user", "content": "What is dependency injection in Python?"},
    ],
    # Medium conversation (repetitive context, tests deduplication & cleaning)
    [
        {"role": "system", "content": "You are a database consultant."},
        {"role": "user", "content": "Here is our schema:\n\nCREATE TABLE users (id INT, name TEXT);\n---\n---\nCREATE TABLE orders (id INT, user_id INT);\n\n\n\nHow do we index this?"},
        {"role": "assistant", "content": "Create a foreign key index on orders(user_id)."},
        {"role": "user", "content": "How do we index this?\n\nCREATE TABLE orders (id INT, user_id INT);"},
    ],
    # Large multi-turn conversation (requires TF-IDF sentence importance compression)
    [
        {"role": "system", "content": "You are an enterprise AI architecture specialist."},
        {"role": "user", "content": "In our current infrastructure, we have legacy microservices running across three AWS regions with high latency. " * 15},
        {"role": "assistant", "content": "I recommend deploying an API gateway layer with regional edge caching and DynamoDB global tables."},
        {"role": "user", "content": "We also have intermittent network timeouts between our US-East and EU-West clusters during peak hours. " * 20},
        {"role": "assistant", "content": "You should configure mutual TLS with connection pooling and circuit breakers."},
        {"role": "user", "content": "What specific timeout thresholds and backoff retry algorithms should we configure for the inter-region RPCs?"},
    ],
]


def run_compression_benchmark() -> Dict:
    results_by_mode = {}

    for mode in ["minimal", "smart", "aggressive"]:
        mode_latencies = []
        original_tokens_list = []
        compressed_tokens_list = []
        tokens_saved_list = []

        for convo in SAMPLE_CONVERSATIONS:
            start = time.perf_counter()
            _, stats = compress(convo, model="gpt-4o", max_tokens=100, mode=mode)
            latency = (time.perf_counter() - start) * 1000

            mode_latencies.append(latency)
            original_tokens_list.append(stats["original_tokens"])
            compressed_tokens_list.append(stats["compressed_tokens"])
            tokens_saved_list.append(stats["tokens_saved"])

        total_orig = sum(original_tokens_list)
        total_comp = sum(compressed_tokens_list)
        total_saved = sum(tokens_saved_list)
        overall_reduction = ((total_orig - total_comp) / max(total_orig, 1)) * 100

        results_by_mode[mode] = {
            "avg_latency_ms": statistics.mean(mode_latencies),
            "p95_latency_ms": sorted(mode_latencies)[int(len(mode_latencies) * 0.95)],
            "total_original_tokens": total_orig,
            "total_compressed_tokens": total_comp,
            "total_tokens_saved": total_saved,
            "reduction_pct": overall_reduction,
        }

    return results_by_mode


def main():
    print("🔬 Running OptiLLM Context Compression Benchmark...")
    results = run_compression_benchmark()

    print("\nCompression Performance by Mode:")
    print("  Mode       | Reduction % | Tokens Saved | Avg Latency | P95 Latency")
    print("  ───────────┼─────────────┼──────────────┼─────────────┼────────────")
    for mode, stats in results.items():
        print(f"  {mode.ljust(10)} |   {stats['reduction_pct']:5.1f}%    |   {stats['total_tokens_saved']:6d}     |   {stats['avg_latency_ms']:5.2f}ms   |   {stats['p95_latency_ms']:5.2f}ms")
    print()


if __name__ == "__main__":
    main()
