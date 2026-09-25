"""
Prometheus Observability Metrics Engine
Defines request counters, latency histograms, cache metrics, and cost savings counters.
Provides exposition text renderer for GET /metrics endpoint.
"""

import logging
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger("optillm.core.metrics")

# Gateway Prometheus Metrics Definitions
HTTP_REQUESTS_TOTAL = Counter(
    "optillm_http_requests_total",
    "Total HTTP requests received by OptiLLM Gateway",
    ["method", "endpoint", "status_code"],
)

HTTP_LATENCY_SECONDS = Histogram(
    "optillm_http_request_duration_seconds",
    "HTTP request latency histogram in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

CACHE_HITS_TOTAL = Counter(
    "optillm_cache_hits_total",
    "Total semantic cache hits",
    ["namespace"],
)

CACHE_MISSES_TOTAL = Counter(
    "optillm_cache_misses_total",
    "Total semantic cache misses",
    ["namespace"],
)

TOKENS_SAVED_TOTAL = Counter(
    "optillm_tokens_saved_total",
    "Total tokens saved via context compression and cache hits",
)

COST_SAVINGS_USD_TOTAL = Counter(
    "optillm_cost_savings_usd_total",
    "Total USD cost saved via model routing, compression, and cache hits",
)

PROVIDER_CALLS_TOTAL = Counter(
    "optillm_provider_calls_total",
    "Total requests dispatched to downstream LLM providers",
    ["provider", "model"],
)

CIRCUIT_BREAKER_STATE = Gauge(
    "optillm_circuit_breaker_state",
    "Circuit breaker status (0=closed, 1=open, 2=half-open)",
    ["provider"],
)

RATE_LIMIT_EXCEEDED_TOTAL = Counter(
    "optillm_rate_limit_exceeded_total",
    "Total requests rejected due to RPM, TPM, or auto-blocklist rate limits",
    ["limit_type"],
)

DAILY_BUDGET_UTILIZATION = Gauge(
    "optillm_daily_budget_utilization_ratio",
    "Current spend as a fraction of daily budget limit (0.0 to 1.0+)",
)


def record_request_metrics(method: str, endpoint: str, status_code: int, duration_seconds: float):
    """Record HTTP request counter and latency histogram metrics."""
    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    HTTP_LATENCY_SECONDS.labels(endpoint=endpoint).observe(duration_seconds)


def render_metrics() -> tuple[bytes, str]:
    """
    Renders live metrics in standard Prometheus text format.
    Returns (content_bytes, content_type_header).
    """
    return generate_latest(), CONTENT_TYPE_LATEST
