#!/usr/bin/env python3
"""
OptiLLM Benchmark: Direct Provider vs OptiLLM Gateway
Measures latency, token usage, cost, and cache effectiveness.

Usage:
    python benchmarks/benchmark_gateway.py
    python benchmarks/benchmark_gateway.py --url http://localhost:8000 --runs 10

NOTE: This benchmark requires a valid OpenAI API key in .env.
      Results are measured, NOT invented.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import httpx
except ImportError:
    print("pip install httpx")
    sys.exit(1)

BENCHMARK_PROMPTS = [
    "What is 15% of 240?",
    "Summarize the water cycle in one sentence.",
    "What is the capital of Japan?",
    "What does HTTP stand for?",
    "Convert 100 Fahrenheit to Celsius.",
]

@dataclass
class RequestResult:
    latency_ms: int
    tokens_input: int
    tokens_output: int
    cost_usd: float
    cache_hit: bool
    compressed: bool
    routed: bool
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    runs: int
    results: List[RequestResult] = field(default_factory=list)

    def avg_latency(self) -> float:
        lats = [r.latency_ms for r in self.results if not r.error]
        return statistics.mean(lats) if lats else 0.0

    def p95_latency(self) -> float:
        lats = sorted(r.latency_ms for r in self.results if not r.error)
        if not lats:
            return 0.0
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.results if not r.error)

    def cache_hit_rate(self) -> float:
        valid = [r for r in self.results if not r.error]
        if not valid:
            return 0.0
        return sum(1 for r in valid if r.cache_hit) / len(valid)

    def total_tokens(self) -> int:
        return sum(r.tokens_input + r.tokens_output for r in self.results if not r.error)

    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.error) / len(self.results)


def run_optillm_benchmark(base_url: str, runs: int, api_key: Optional[str] = None) -> BenchmarkReport:
    """Benchmark OptiLLM gateway."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    report = BenchmarkReport(runs=runs)
    client = httpx.Client(base_url=base_url, timeout=60.0, headers=headers)

    for i in range(runs):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
        payload = {
            "model": "gpt-4o",  # Request expensive model — let router downgrade
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        }

        start = time.time()
        try:
            r = client.post("/v1/chat/completions", json=payload)
            elapsed = int((time.time() - start) * 1000)

            if r.status_code == 200:
                data = r.json()
                meta = data.get("optillm_metadata", {})
                usage = data.get("usage", {})
                report.results.append(RequestResult(
                    latency_ms=elapsed,
                    tokens_input=usage.get("prompt_tokens", 0),
                    tokens_output=usage.get("completion_tokens", 0),
                    cost_usd=meta.get("cost_usd", 0.0),
                    cache_hit=meta.get("cache_hit", False),
                    compressed=meta.get("compressed", False),
                    routed=meta.get("routed", False),
                ))
            else:
                report.results.append(RequestResult(
                    latency_ms=elapsed, tokens_input=0, tokens_output=0,
                    cost_usd=0.0, cache_hit=False, compressed=False, routed=False,
                    error=f"HTTP {r.status_code}",
                ))
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            report.results.append(RequestResult(
                latency_ms=elapsed, tokens_input=0, tokens_output=0,
                cost_usd=0.0, cache_hit=False, compressed=False, routed=False,
                error=str(e),
            ))

        sys.stdout.write(f"\r  Progress: {i+1}/{runs}")
        sys.stdout.flush()

    print()
    client.close()
    return report


def print_report(report: BenchmarkReport, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Runs:           {report.runs}")
    print(f"  Errors:         {report.error_rate()*100:.1f}%")
    print(f"  Avg Latency:    {report.avg_latency():.0f}ms")
    print(f"  P95 Latency:    {report.p95_latency():.0f}ms")
    print(f"  Total Tokens:   {report.total_tokens():,}")
    print(f"  Total Cost:     ${report.total_cost():.6f}")
    print(f"  Cache Hit Rate: {report.cache_hit_rate()*100:.1f}%")

    routed = sum(1 for r in report.results if r.routed and not r.error)
    compressed = sum(1 for r in report.results if r.compressed and not r.error)
    print(f"  Routed:         {routed}/{report.runs} requests downgraded to cheaper model")
    print(f"  Compressed:     {compressed}/{report.runs} requests had prompt compressed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiLLM Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--runs", type=int, default=10, help="Number of requests")
    parser.add_argument("--api-key", default=None, help="API key")
    args = parser.parse_args()

    print("\n🔬 OptiLLM Gateway Benchmark")
    print(f"   URL:  {args.url}")
    print(f"   Runs: {args.runs}")
    print("\n  Running benchmark...")

    report = run_optillm_benchmark(args.url, args.runs, args.api_key)
    print_report(report, "OptiLLM Gateway Results")

    if report.cache_hit_rate() > 0:
        cache_savings = report.total_cost() * report.cache_hit_rate()
        print(f"\n  Estimated Cache Savings: ${cache_savings:.6f}")

    print("\n  NOTE: Run with --runs 20+ and real API keys for meaningful results.")
    print("        First run will always miss cache; subsequent runs will hit it.\n")
