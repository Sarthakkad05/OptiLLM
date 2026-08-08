#!/usr/bin/env python3
"""
OptiLLM Production Smoke Test
Verifies core functionality of a running OptiLLM gateway.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --url http://my-server:8000
    python scripts/smoke_test.py --api-key sk-optillm-my-key

Exit codes:
    0 — All tests passed
    1 — One or more tests failed
"""

import argparse
import json
import sys
import time
from typing import Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def check(name: str, passed: bool, detail: str = ""):
    icon = PASS if passed else FAIL
    suffix = f"  — {detail}" if detail else ""
    print(f"  {icon} {name}{suffix}")
    return passed


def run_smoke_tests(base_url: str, api_key: Optional[str] = None) -> bool:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = httpx.Client(base_url=base_url, timeout=30.0, headers=headers)
    results = []

    print(f"\n{'='*60}")
    print(f"  OptiLLM Smoke Test — {base_url}")
    print(f"{'='*60}\n")

    # ── Test 1: Health ─────────────────────────────────────────────────────────
    print("1. Health Check")
    try:
        r = client.get("/health")
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "ok"
        results.append(check("GET /health", ok, f"status={data.get('status')} db={data.get('database')}"))
    except Exception as e:
        results.append(check("GET /health", False, str(e)))

    # ── Test 2: Readiness ──────────────────────────────────────────────────────
    print("\n2. Readiness Check")
    try:
        r = client.get("/ready")
        ok = r.status_code in (200, 503)  # 503 is ok if no providers configured
        data = r.json()
        results.append(check("GET /ready", ok, f"status={data.get('status')}"))
        if r.status_code == 503:
            print(f"     {WARN} No providers configured — mock mode will be used")
    except Exception as e:
        results.append(check("GET /ready", False, str(e)))

    # ── Test 3: Model Registry ─────────────────────────────────────────────────
    print("\n3. Model Registry")
    try:
        r = client.get("/v1/models")
        data = r.json()
        ok = r.status_code == 200 and data.get("object") == "list"
        model_count = len(data.get("data", []))
        results.append(check("GET /v1/models", ok, f"{model_count} models registered"))
    except Exception as e:
        results.append(check("GET /v1/models", False, str(e)))

    # ── Test 4: Analytics ──────────────────────────────────────────────────────
    print("\n4. Analytics")
    try:
        r = client.get("/api/v1/analytics")
        ok = r.status_code == 200
        data = r.json()
        total = data.get("summary", {}).get("total_requests", 0)
        results.append(check("GET /api/v1/analytics", ok, f"total_requests={total}"))
    except Exception as e:
        results.append(check("GET /api/v1/analytics", False, str(e)))

    # ── Test 5: Provider Status ────────────────────────────────────────────────
    print("\n5. Provider Status")
    try:
        r = client.get("/api/v1/providers/status")
        ok = r.status_code == 200
        data = r.json()
        available = data.get("available_providers_count", 0)
        results.append(check("GET /api/v1/providers/status", ok, f"available_providers={available}"))
    except Exception as e:
        results.append(check("GET /api/v1/providers/status", False, str(e)))

    # ── Test 6: Chat Completion ────────────────────────────────────────────────
    print("\n6. Chat Completion")
    try:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Respond with exactly: SMOKE_TEST_OK"}],
            "max_tokens": 20,
        }
        r = client.post("/v1/chat/completions", json=payload)
        data = r.json()
        ok = r.status_code == 200 and "choices" in data
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
        meta = data.get("optillm_metadata", {})
        results.append(check(
            "POST /v1/chat/completions",
            ok,
            f"content={repr(content)} cache_hit={meta.get('cache_hit')} cost=${meta.get('cost_usd', 0):.6f}"
        ))
    except Exception as e:
        results.append(check("POST /v1/chat/completions", False, str(e)))

    # ── Test 7: Semantic Cache ─────────────────────────────────────────────────
    print("\n7. Semantic Cache")
    try:
        # Send same request twice — second should be a cache hit
        cache_payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "max_tokens": 30,
        }
        r1 = client.post("/v1/chat/completions", json=cache_payload)
        time.sleep(0.5)
        r2 = client.post("/v1/chat/completions", json=cache_payload)

        ok1 = r1.status_code == 200
        ok2 = r2.status_code == 200

        hit1 = r1.json().get("optillm_metadata", {}).get("cache_hit", False) if ok1 else False
        hit2 = r2.json().get("optillm_metadata", {}).get("cache_hit", False) if ok2 else False

        results.append(check("First request (cache miss)", ok1, f"cache_hit={hit1}"))
        results.append(check("Second request (cache hit)", ok2 and hit2, f"cache_hit={hit2}"))
    except Exception as e:
        results.append(check("Semantic Cache", False, str(e)))

    # ── Test 8: Streaming ─────────────────────────────────────────────────────
    print("\n8. Streaming")
    try:
        stream_payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Count to 3."}],
            "stream": True,
            "max_tokens": 30,
        }
        chunks = []
        with client.stream("POST", "/v1/chat/completions", json=stream_payload) as r:
            ok = r.status_code == 200
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        chunks.append(chunk)
                    except Exception:
                        pass
        results.append(check("POST /v1/chat/completions (stream=true)", ok and len(chunks) > 0, f"{len(chunks)} chunks received"))
    except Exception as e:
        results.append(check("Streaming", False, str(e)))

    # ── Test 9: Prometheus Metrics ─────────────────────────────────────────────
    print("\n9. Prometheus Metrics")
    try:
        r = client.get("/metrics")
        ok = r.status_code == 200 and "optillm_" in r.text
        metric_count = r.text.count("# HELP optillm_")
        results.append(check("GET /metrics", ok, f"{metric_count} OptiLLM metrics exposed"))
    except Exception as e:
        results.append(check("GET /metrics", False, str(e)))

    # ── Test 10: Error Handling ────────────────────────────────────────────────
    print("\n10. Error Handling")
    try:
        r = client.post("/v1/chat/completions", json={"invalid": "request"})
        ok = r.status_code == 422  # Validation error
        results.append(check("Invalid request returns 422", ok, f"status={r.status_code}"))
    except Exception as e:
        results.append(check("Error handling", False, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{'='*60}")
    if passed == total:
        print(f"  {PASS} All {total} smoke tests PASSED")
        print(f"\n  OptiLLM gateway is production-ready.")
    else:
        failed = total - passed
        print(f"  {FAIL} {failed}/{total} smoke tests FAILED")
    print(f"{'='*60}\n")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiLLM Smoke Tests")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL")
    parser.add_argument("--api-key", default=None, help="Bearer API key")
    args = parser.parse_args()

    success = run_smoke_tests(args.url, args.api_key)
    sys.exit(0 if success else 1)
