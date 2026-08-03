"""
OptiLLM Developer CLI Interface
Command-line utility for managing gateway execution, checking health,
inspecting request logs, fetching metrics, and clearing cache.
"""

import argparse
import sys
import httpx


def main():
    parser = argparse.ArgumentParser(description="OptiLLM Gateway Developer CLI")
    subparsers = parser.add_subparsers(dest="command", help="CLI Command")

    # Command: start
    start_parser = subparsers.add_parser("start", help="Start OptiLLM Gateway server")
    start_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    start_parser.add_argument("--port", type=int, default=8000, help="Port number")

    # Command: status
    subparsers.add_parser("status", help="Check gateway health and provider status")

    # Command: logs
    subparsers.add_parser("logs", help="Fetch recent gateway request logs")

    # Command: metrics
    subparsers.add_parser("metrics", help="Fetch live Prometheus exposition metrics")

    # Command: cache-clear
    subparsers.add_parser("cache-clear", help="Clear semantic cache entries")

    args = parser.parse_args()

    if not args.command or args.command == "status":
        check_status()
    elif args.command == "start":
        run_server(args.host, args.port)
    elif args.command == "logs":
        show_logs()
    elif args.command == "metrics":
        show_metrics()
    elif args.command == "cache-clear":
        clear_cache()
    else:
        parser.print_help()


def check_status(base_url: str = "http://localhost:8000"):
    try:
        res = httpx.get(f"{base_url}/health", timeout=5.0)
        print("🟢 OptiLLM Gateway Status: ONLINE")
        print(f"Health Response: {res.json()}")

        p_res = httpx.get(f"{base_url}/api/v1/providers/status", timeout=5.0)
        if p_res.status_code == 200:
            print("\nLLM Providers Health:")
            for p in p_res.json().get("providers", []):
                status_icon = "✅" if p.get("healthy") else "❌"
                print(f"  {status_icon} {p.get('name')}: {p.get('status')}")
    except Exception as e:
        print(f"🔴 OptiLLM Gateway Status: UNREACHABLE ({e})")


def run_server(host: str, port: int):
    import uvicorn
    print(f"🚀 Starting OptiLLM Gateway on http://{host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)


def show_logs(base_url: str = "http://localhost:8000"):
    try:
        res = httpx.get(f"{base_url}/api/v1/analytics", timeout=5.0)
        data = res.json()
        recent = data.get("recent_requests", [])
        print(f"📋 Recent Request Logs ({len(recent)} entries):")
        for r in recent[:10]:
            print(f"  [{r.get('timestamp')}] {r.get('model_requested')} -> {r.get('model_used')} | Latency: {r.get('latency_ms')}ms | Cost: ${r.get('cost_usd'):.6f}")
    except Exception as e:
        print(f"Error fetching logs: {e}")


def show_metrics(base_url: str = "http://localhost:8000"):
    try:
        res = httpx.get(f"{base_url}/metrics", timeout=5.0)
        print(res.text)
    except Exception as e:
        print(f"Error fetching metrics: {e}")


def clear_cache(base_url: str = "http://localhost:8000"):
    try:
        res = httpx.delete(f"{base_url}/api/v1/cache/clear", timeout=5.0)
        print(res.json().get("message", "Cache cleared."))
    except Exception as e:
        print(f"Error clearing cache: {e}")


if __name__ == "__main__":
    main()
