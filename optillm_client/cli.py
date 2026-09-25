"""
OptiLLM Developer CLI Interface
Command-line utility for interacting with and managing the OptiLLM Gateway:
- Chat completions and streaming directly from CLI
- Model routing explanation for given prompts
- Real-time quality-cost analytics and spend reporting
- Semantic cache inspection, warming, and invalidation
- API key lifecycle management (create, list, revoke)
- Request feedback submission
- Gateway health status and live Prometheus metrics
"""

import argparse
import json
import os
import sys
from typing import Optional

try:
    import httpx
except ImportError:
    print("Error: httpx is required for the OptiLLM CLI. Run: pip install httpx")
    sys.exit(1)


def _get_base_url(args) -> str:
    url = getattr(args, "url", None) or os.environ.get("OPTILLM_URL", "http://localhost:8000")
    return url.rstrip("/")


def _get_api_key(args) -> str:
    return getattr(args, "api_key", None) or os.environ.get("OPTILLM_API_KEY", "sk-optillm-dev-key")


def _headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def main():
    parser = argparse.ArgumentParser(
        prog="optillm",
        description="OptiLLM AI Gateway — Developer & Admin CLI",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("OPTILLM_URL", "http://localhost:8000"),
        help="OptiLLM gateway URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPTILLM_API_KEY", "sk-optillm-dev-key"),
        help="OptiLLM API key",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: status
    subparsers.add_parser("status", help="Check gateway liveness, readiness, and provider circuits")

    # Command: start
    start_parser = subparsers.add_parser("start", help="Start the local OptiLLM Gateway server")
    start_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    start_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    # Command: chat
    chat_parser = subparsers.add_parser("chat", help="Send a prompt through the optimization pipeline")
    chat_parser.add_argument("prompt", help="User prompt to send")
    chat_parser.add_argument("--model", default="gpt-4o", help="Requested model (default: gpt-4o)")
    chat_parser.add_argument("--stream", action="store_true", help="Stream the response via SSE")
    chat_parser.add_argument("--bypass-cache", action="store_true", help="Bypass semantic cache")
    chat_parser.add_argument("--tag", default="cli", help="Metadata tag for analytics")

    # Command: explain
    explain_parser = subparsers.add_parser("explain", help="Explain routing decision and complexity for a prompt")
    explain_parser.add_argument("prompt", help="Prompt to analyze")
    explain_parser.add_argument("--model", default="gpt-4o", help="Target model (default: gpt-4o)")

    # Command: analytics
    analytics_parser = subparsers.add_parser("analytics", help="View spend, savings, and quality-cost analytics")
    analytics_parser.add_argument("--days", type=int, default=7, help="Analysis window in days (default: 7)")
    analytics_parser.add_argument("--tradeoff", action="store_true", help="Show quality-cost tradeoff recommendation")

    # Command: cache
    cache_parser = subparsers.add_parser("cache", help="Manage semantic cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_action", help="Cache action")
    cache_sub.add_parser("info", help="Display cache status and statistics")
    cache_sub.add_parser("clear", help="Invalidate and wipe semantic cache entries")
    cache_sub.add_parser("warm", help="Pre-warm semantic cache from DB entries")

    # Command: feedback
    fb_parser = subparsers.add_parser("feedback", help="Submit quality rating on a previous request")
    fb_parser.add_argument("request_id", help="Request ID from previous completion")
    fb_parser.add_argument("--rating", type=int, choices=[1, 0, -1], required=True, help="Rating: 1=good, 0=neutral, -1=bad")
    fb_parser.add_argument("--issue", choices=["wrong", "slow", "incomplete", "hallucinated", "other"], help="Issue category")
    fb_parser.add_argument("--notes", help="Optional detailed notes")

    # Command: keys
    keys_parser = subparsers.add_parser("keys", help="Manage API keys")
    keys_sub = keys_parser.add_subparsers(dest="keys_action", help="Keys action")
    keys_sub.add_parser("list", help="List active API keys")
    create_key = keys_sub.add_parser("create", help="Generate a new API key")
    create_key.add_argument("--name", required=True, help="Key identifier / name")
    create_key.add_argument("--rpm", type=int, help="Requests per minute rate limit")
    create_key.add_argument("--tpm", type=int, help="Tokens per minute rate limit")
    delete_key = keys_sub.add_parser("delete", help="Revoke an API key")
    delete_key.add_argument("key_id", help="Key record ID to delete")

    # Command: logs
    subparsers.add_parser("logs", help="Fetch recent gateway request logs")

    # Command: metrics
    subparsers.add_parser("metrics", help="Fetch live Prometheus exposition metrics")

    args = parser.parse_args()

    if not args.command or args.command == "status":
        cmd_status(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "analytics":
        cmd_analytics(args)
    elif args.command == "cache":
        cmd_cache(args)
    elif args.command == "feedback":
        cmd_feedback(args)
    elif args.command == "keys":
        cmd_keys(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    else:
        parser.print_help()


# ── Command Implementations ───────────────────────────────────────────────────


def cmd_status(args):
    base_url = _get_base_url(args)
    try:
        res = httpx.get(f"{base_url}/health/ready", timeout=5.0)
        data = res.json()
        status = data.get("status", "unknown").upper()
        icon = "🟢" if status == "READY" else ("🟡" if status == "DEGRADED" else "🔴")

        print(f"\n{icon} OptiLLM Gateway Status: {status}")
        print(f"   Version : {data.get('version')}")
        print(f"   Uptime  : {data.get('uptime_seconds')}s\n")

        checks = data.get("checks", {})
        print("Subsystems:")
        for name, detail in checks.items():
            if isinstance(detail, dict):
                st = detail.get("status", "ok")
                lat = f" ({detail.get('latency_ms')}ms)" if "latency_ms" in detail else ""
                st_icon = "✅" if st in ("ok", "not_configured") else "❌"
                print(f"  {st_icon} {name.ljust(16)}: {st}{lat}")

        # Show provider status
        p_res = httpx.get(f"{base_url}/api/v1/providers/status", timeout=5.0)
        if p_res.status_code == 200:
            p_data = p_res.json()
            print("\nAI Providers:")
            for p in p_data.get("providers", []):
                p_icon = "✅" if p.get("healthy") else "❌"
                print(f"  {p_icon} {p.get('name').ljust(12)}: {p.get('status')}")
        print()
    except Exception as e:
        print(f"🔴 OptiLLM Gateway UNREACHABLE at {base_url} ({e})")


def cmd_start(args):
    import uvicorn
    print(f"🚀 Starting OptiLLM Gateway on http://{args.host}:{args.port}...")
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)


def cmd_chat(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": args.stream,
        "optillm": {
            "bypass_cache": args.bypass_cache,
        },
    }

    if args.stream:
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={**_headers(api_key), "x-optillm-tag": args.tag},
            ) as response:
                if response.status_code != 200:
                    print(f"Error ({response.status_code}): {response.read().decode()}")
                    return
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            sys.stdout.write(token)
                            sys.stdout.flush()
                        except Exception:
                            pass
                print()
    else:
        try:
            res = httpx.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers={**_headers(api_key), "x-optillm-tag": args.tag},
                timeout=60.0,
            )
            if res.status_code != 200:
                print(f"Error ({res.status_code}): {res.text}")
                return
            data = res.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            meta = data.get("optillm_metadata", {})

            print(f"\n{content}\n")
            print("─" * 60)
            print(f"Model used    : {meta.get('model_used')} (requested: {meta.get('model_requested')})")
            print(f"Cache hit     : {meta.get('cache_hit')}")
            print(f"Latency       : {meta.get('latency_ms')}ms")
            print(f"Cost          : ${meta.get('cost_usd', 0.0):.6f} (saved: ${meta.get('savings_usd', 0.0):.6f})")
            if meta.get("complexity"):
                print(f"Complexity    : {meta.get('complexity')}")
            print("─" * 60)
        except Exception as e:
            print(f"Request failed: {e}")


def cmd_explain(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)

    try:
        res = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": args.prompt}],
                "optillm": {"bypass_cache": True},
            },
            headers=_headers(api_key),
            timeout=30.0,
        )
        if res.status_code != 200:
            print(f"Error ({res.status_code}): {res.text}")
            return
        data = res.json()
        meta = data.get("optillm_metadata", {})
        print("\n🔍 OptiLLM Routing Analysis:")
        print(f"  Prompt               : \"{args.prompt[:60]}...\"")
        print(f"  Requested Model      : {meta.get('model_requested')}")
        print(f"  Routed Model         : {meta.get('model_used')}")
        print(f"  Assessed Complexity  : {meta.get('complexity')}")
        print(f"  Routing Reason       : {meta.get('routing_reason')}")
        print(f"  Estimated Cost       : ${meta.get('cost_usd', 0.0):.6f}")
        print(f"  Savings Achieved     : ${meta.get('savings_usd', 0.0):.6f}\n")
    except Exception as e:
        print(f"Analysis failed: {e}")


def cmd_analytics(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)

    try:
        if args.tradeoff:
            res = httpx.get(
                f"{base_url}/api/v1/analytics/quality-cost-tradeoff",
                headers=_headers(api_key),
                timeout=10.0,
            )
            if res.status_code == 200:
                data = res.json()
                print("\n📊 Quality vs Cost Tradeoff Analysis:")
                print(f"  Recommendation: {data.get('recommendation', 'N/A')}\n")
                models = data.get("by_model", [])
                for m in models:
                    print(f"  • {m.get('model')}:")
                    print(f"      Avg Cost/Req  : ${m.get('avg_cost_per_request', 0.0):.6f}")
                    print(f"      Avg Quality   : {m.get('avg_quality_score', 0.0):.2f}")
                    print(f"      Quality/Dollar: {m.get('quality_per_dollar', 0.0):.1f}")
                print()
                return

        res = httpx.get(
            f"{base_url}/api/v1/analytics?days={args.days}",
            headers=_headers(api_key),
            timeout=10.0,
        )
        if res.status_code != 200:
            print(f"Error ({res.status_code}): {res.text}")
            return
        data = res.json()
        summary = data.get("summary", {})
        print(f"\n📈 OptiLLM Gateway Analytics (Last {args.days} Days):")
        print(f"  Total Requests     : {summary.get('total_requests', 0)}")
        print(f"  Cache Hit Rate     : {summary.get('cache_hit_rate_pct', 0.0):.1f}%")
        print(f"  Total Spend        : ${summary.get('total_cost_usd', 0.0):.4f}")
        print(f"  Total Saved        : ${summary.get('total_savings_usd', 0.0):.4f}")
        print(f"  Tokens Saved       : {summary.get('total_tokens_saved', 0):,}")
        print(f"  Avg Latency        : {summary.get('avg_latency_ms', 0.0):.1f}ms\n")
    except Exception as e:
        print(f"Analytics query failed: {e}")


def cmd_cache(args):
    base_url = _get_base_url(args)
    action = args.cache_action or "info"

    if action == "info":
        try:
            res = httpx.get(f"{base_url}/api/v1/cache/info", timeout=5.0)
            data = res.json()
            print("\n💾 Semantic Cache Info:")
            print(f"  Total Cached Responses : {data.get('total_cached_responses')}")
            print(f"  Redis Available        : {data.get('redis_available')}")
            print(f"  Namespace              : {data.get('cache_namespace')}")
            print(f"  Default TTL            : {data.get('cache_ttl_seconds')}s\n")
        except Exception as e:
            print(f"Error: {e}")
    elif action == "clear":
        try:
            res = httpx.delete(f"{base_url}/api/v1/cache/clear", timeout=5.0)
            print(f"✅ {res.json().get('message', 'Cache cleared.')}")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    elif action == "warm":
        try:
            res = httpx.post(f"{base_url}/api/v1/cache/warm", timeout=10.0)
            print(f"✅ {res.json().get('message', 'Cache warmed.')}")
        except Exception as e:
            print(f"Error warming cache: {e}")


def cmd_feedback(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)

    try:
        res = httpx.post(
            f"{base_url}/api/v1/feedback/{args.request_id}",
            json={
                "rating": args.rating,
                "issue": args.issue,
                "notes": args.notes,
            },
            headers=_headers(api_key),
            timeout=10.0,
        )
        if res.status_code == 200:
            print(f"✅ Feedback logged for request {args.request_id}.")
        else:
            print(f"Error ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"Feedback submission failed: {e}")


def cmd_keys(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    action = args.keys_action or "list"

    if action == "list":
        try:
            res = httpx.get(f"{base_url}/v1/keys", headers=_headers(api_key), timeout=5.0)
            if res.status_code == 200:
                keys = res.json()
                print(f"\n🔑 Active API Keys ({len(keys)}):")
                for k in keys:
                    print(f"  • {k.get('name', 'unnamed')} (ID: {k.get('id')}) | RPM: {k.get('rpm_limit')} | TPM: {k.get('tpm_limit')}")
                print()
            else:
                print(f"Error ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"Error: {e}")
    elif action == "create":
        try:
            res = httpx.post(
                f"{base_url}/v1/keys",
                json={"name": args.name, "rpm_limit": args.rpm, "tpm_limit": args.tpm},
                headers=_headers(api_key),
                timeout=5.0,
            )
            if res.status_code == 200:
                data = res.json()
                print("\n✅ New API Key Created:")
                print(f"  Key: {data.get('key')}")
                print(f"  ID : {data.get('id')}")
                print("  ⚠️  Save this key now — it cannot be shown again.\n")
            else:
                print(f"Error ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"Error creating key: {e}")
    elif action == "delete":
        try:
            res = httpx.delete(f"{base_url}/v1/keys/{args.key_id}", headers=_headers(api_key), timeout=5.0)
            print(f"✅ Key {args.key_id} deleted.")
        except Exception as e:
            print(f"Error deleting key: {e}")


def cmd_logs(args):
    base_url = _get_base_url(args)
    api_key = _get_api_key(args)
    try:
        res = httpx.get(f"{base_url}/api/v1/analytics", headers=_headers(api_key), timeout=5.0)
        data = res.json()
        recent = data.get("recent_requests", [])
        print(f"\n📋 Recent Gateway Request Logs ({len(recent)} entries):")
        for r in recent[:10]:
            print(f"  [{r.get('timestamp')}] {r.get('model_requested')} -> {r.get('model_used')} | Latency: {r.get('latency_ms')}ms | Cost: ${r.get('cost_usd', 0.0):.6f}")
        print()
    except Exception as e:
        print(f"Error fetching logs: {e}")


def cmd_metrics(args):
    base_url = _get_base_url(args)
    try:
        res = httpx.get(f"{base_url}/metrics", timeout=5.0)
        print(res.text)
    except Exception as e:
        print(f"Error fetching metrics: {e}")


if __name__ == "__main__":
    main()
