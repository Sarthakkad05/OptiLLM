#!/usr/bin/env python3
"""
OptiLLM Semantic Cache Benchmark
Measures cache lookup latency, cache insertion speed, and hit rates at varying similarity thresholds.

Usage:
    python benchmarks/bench_cache.py
"""

import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.engine.cache import check_cache, insert_cache
from app.engine.faiss_store import rebuild_from_entries

TEST_QUERIES = [
    ("What is the speed of light in vacuum?", "The speed of light in vacuum is approximately 299,792,458 meters per second."),
    ("Explain the difference between TCP and UDP.", "TCP is connection-oriented and guarantees delivery; UDP is connectionless and prioritized for speed."),
    ("How does binary search work?", "Binary search divides a sorted array in half repeatedly until the target is found in O(log n) time."),
    ("What is gradient descent?", "Gradient descent is an optimization algorithm that iteratively steps in the direction of steepest descent."),
    ("Define Python list comprehension.", "A concise syntactic construct in Python to create new lists from existing iterables."),
]

PARAPHRASED_QUERIES = [
    ("Tell me the vacuum velocity of light.", 0),
    ("Compare TCP vs UDP protocols.", 1),
    ("How does binary search find elements?", 2),
    ("What does the gradient descent algorithm do?", 3),
    ("Explain list comprehensions in Python language.", 4),
]


def run_cache_benchmark(thresholds: List[float] = [0.80, 0.85, 0.90, 0.95]) -> Dict:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    rebuild_from_entries([])

    # 1. Warm cache and measure insertion latency
    insert_latencies_ms = []
    for query, response in TEST_QUERIES:
        start = time.perf_counter()
        insert_cache(
            messages=[{"role": "user", "content": query}],
            response_text=response,
            model="gpt-4o",
            tokens_input=15,
            tokens_output=25,
            db=db,
        )
        insert_latencies_ms.append((time.perf_counter() - start) * 1000)

    # 2. Benchmark exact match lookup
    exact_latencies_ms = []
    for query, _ in TEST_QUERIES:
        start = time.perf_counter()
        hit = check_cache(
            messages=[{"role": "user", "content": query}],
            similarity_threshold=0.90,
            db=db,
        )
        exact_latencies_ms.append((time.perf_counter() - start) * 1000)
        assert hit is not None

    # 3. Benchmark semantic paraphrased match across thresholds
    threshold_results = {}
    for th in thresholds:
        hits = 0
        latencies = []
        for paraphrased, expected_idx in PARAPHRASED_QUERIES:
            start = time.perf_counter()
            hit = check_cache(
                messages=[{"role": "user", "content": paraphrased}],
                similarity_threshold=th,
                db=db,
            )
            latencies.append((time.perf_counter() - start) * 1000)
            if hit is not None:
                hits += 1

        threshold_results[th] = {
            "hit_rate": hits / len(PARAPHRASED_QUERIES),
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        }

    return {
        "avg_insert_latency_ms": statistics.mean(insert_latencies_ms),
        "avg_exact_lookup_ms": statistics.mean(exact_latencies_ms),
        "threshold_results": threshold_results,
    }


def main():
    print("🔬 Running OptiLLM Semantic Cache Benchmark...")
    results = run_cache_benchmark()

    print("\nCache Latency Results:")
    print(f"  • Avg Insert Latency : {results['avg_insert_latency_ms']:.2f}ms")
    print(f"  • Avg Exact Lookup   : {results['avg_exact_lookup_ms']:.2f}ms\n")

    print("Similarity Threshold Evaluation:")
    print("  Threshold | Hit Rate | Avg Latency | P95 Latency")
    print("  ──────────┼──────────┼─────────────┼────────────")
    for th, stats in results["threshold_results"].items():
        print(f"    {th:.2f}    |  {stats['hit_rate']*100:5.1f}%  |   {stats['avg_latency_ms']:6.2f}ms  |  {stats['p95_latency_ms']:6.2f}ms")
    print()


if __name__ == "__main__":
    main()
