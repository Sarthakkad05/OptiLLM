"""
demos/base.py
=============
Shared foundation for all OptiLLM demo scenarios.

Provides:
  - DemoStep   : Result of a single gateway call within a scenario
  - DemoResult : Result of a full scenario (one or more steps)
  - call_gateway() : HTTP helper that calls POST /v1/chat/completions
  - check_health() : Pings GET /health
  - fetch_analytics() : Fetches GET /api/v1/analytics
"""

import time
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 60


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class DemoStep:
    """Structured result of a single gateway call."""
    label: str                        # e.g. "Request 1 — Cache Miss"
    prompt: str                       # The user prompt sent
    status: str = "success"           # "success" | "error"
    error: Optional[str] = None

    # OptiLLM optimization flags
    cache_hit: bool = False
    compressed: bool = False
    routed: bool = False

    # Model info
    model_requested: str = ""
    model_used: str = ""
    provider: str = ""

    # Performance
    latency_ms: int = 0
    cost_usd: float = 0.0
    savings_usd: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_saved: int = 0

    # Routing details
    routing_reason: str = ""
    complexity: str = ""

    # Response
    response_preview: str = ""        # First 160 chars

    # Raw metadata (full optillm_metadata dict)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DemoResult:
    """Complete result for a scenario — one or more steps."""
    scenario_key: str
    scenario_name: str
    description: str
    status: str = "success"           # "success" | "error" | "partial"
    steps: List[DemoStep] = field(default_factory=list)

    # Aggregated across all steps
    total_savings_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens_saved: int = 0
    elapsed_ms: int = 0

    # Extra data (used by analytics_summary scenario)
    extra: Dict[str, Any] = field(default_factory=dict)

    def compute_totals(self):
        """Recompute aggregated fields from step data."""
        self.total_savings_usd = sum(s.savings_usd for s in self.steps)
        self.total_cost_usd = sum(s.cost_usd for s in self.steps)
        self.total_tokens_saved = sum(s.tokens_saved for s in self.steps)


# ── HTTP Helpers ───────────────────────────────────────────────────────────────

def check_health() -> bool:
    """Return True if the OptiLLM backend is reachable."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def fetch_analytics() -> Optional[Dict]:
    """Fetch /api/v1/analytics. Returns dict or None on failure."""
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/analytics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_gateway(
    messages: List[Dict],
    model: str = "gpt-4o",
    bypass_cache: bool = False,
    bypass_compression: bool = False,
    bypass_routing: bool = False,
    label: str = "Request",
    prompt_override: str = "",
) -> DemoStep:
    """
    Call POST /v1/chat/completions and return a structured DemoStep.
    Never raises — errors are captured in DemoStep.status / DemoStep.error.
    """
    prompt = prompt_override or _extract_user_prompt(messages)

    payload = {
        "model": model,
        "messages": messages,
        "optillm": {
            "bypass_cache": bypass_cache,
            "bypass_compression": bypass_compression,
            "bypass_routing": bypass_routing,
        },
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        elapsed = int((time.time() - t0) * 1000)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return DemoStep(
            label=label, prompt=prompt, status="error",
            error="Cannot connect to OptiLLM backend (port 8000). Is it running?",
        )
    except requests.exceptions.Timeout:
        return DemoStep(
            label=label, prompt=prompt, status="error",
            error=f"Request timed out after {TIMEOUT}s.",
        )
    except requests.exceptions.HTTPError as e:
        return DemoStep(
            label=label, prompt=prompt, status="error",
            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )
    except Exception as e:
        return DemoStep(
            label=label, prompt=prompt, status="error",
            error=str(e),
        )

    meta = data.get("optillm_metadata", {})
    usage = data.get("usage", {})
    content = ""
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")

    return DemoStep(
        label=label,
        prompt=prompt,
        status="success",
        cache_hit=meta.get("cache_hit", False),
        compressed=meta.get("compressed", False),
        routed=meta.get("routed", False),
        model_requested=meta.get("model_requested", model),
        model_used=meta.get("model_used", "?"),
        provider=meta.get("provider", ""),
        latency_ms=meta.get("latency_ms", elapsed),
        cost_usd=meta.get("cost_usd", 0.0),
        savings_usd=meta.get("savings_usd", 0.0),
        tokens_input=usage.get("prompt_tokens", 0),
        tokens_output=usage.get("completion_tokens", 0),
        tokens_saved=meta.get("tokens_saved", 0),
        routing_reason=meta.get("routing_reason") or "",
        complexity=meta.get("complexity") or "",
        response_preview=content[:160] + ("…" if len(content) > 160 else ""),
        raw_metadata=meta,
    )


def _extract_user_prompt(messages: List[Dict]) -> str:
    """Pull the last user message content for display."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            return content[:120] + ("…" if len(content) > 120 else "")
    return ""
