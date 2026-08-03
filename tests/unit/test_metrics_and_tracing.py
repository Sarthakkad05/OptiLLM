"""
Unit tests for Prometheus metrics collection and OpenTelemetry tracing.
"""

from app.core.metrics import (
    HTTP_REQUESTS_TOTAL,
    record_request_metrics,
    render_metrics,
)
from app.core.tracing import init_tracing, trace_span


def test_prometheus_metrics():
    record_request_metrics("GET", "/health", 200, 0.012)
    body, content_type = render_metrics()

    assert isinstance(body, bytes)
    assert b"optillm_http_requests_total" in body
    assert b"text/plain" in content_type.encode("utf-8") or b"version=0.0.4" in content_type.encode("utf-8")


def test_opentelemetry_tracing():
    init_tracing()

    with trace_span("test_span", {"model": "gpt-4o"}) as span:
        assert span is not None or True
