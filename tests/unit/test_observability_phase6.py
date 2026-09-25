"""
Unit & Integration Test Suite for Phase 6 (Observability & Operations):
- 6.1 Structured JSON Logging & contextvar correlation
- 6.2 OpenTelemetry Distributed Tracing & Span Instrumentation
- 6.3 Cost Attribution, Savings Waterfall & SLA Analytics
- 6.4 Prometheus Metrics & Alerting Configuration Validation
"""

import json
import logging
import os
import yaml
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.logging import (
    JSONFormatter,
    TextFormatter,
    current_request_id,
    log_event,
    setup_logging,
)
from app.core.metrics import (
    CIRCUIT_BREAKER_STATE,
    DAILY_BUDGET_UTILIZATION,
    HTTP_REQUESTS_TOTAL,
    RATE_LIMIT_EXCEEDED_TOTAL,
    record_request_metrics,
    render_metrics,
)
from app.core.tracing import get_tracer, init_tracing, trace_span
from app.db.session import SessionLocal
from app.main import app
from app.services.analytics import get_cost_attribution
from app.services.sla_manager import get_sla_manager


# ── 6.1 Structured JSON Logging Tests ─────────────────────────────────────────

def test_json_formatter_fields():
    formatter = JSONFormatter()
    token = current_request_id.set("req-abc-999")
    try:
        record = logging.LogRecord(
            name="optillm.test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=42,
            msg="User prompt was compressed",
            args=(),
            exc_info=None,
        )
        record.event = "prompt_compressed"
        record.original_tokens = 150
        record.compressed_tokens = 85

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "WARNING"
        assert data["logger"] == "optillm.test"
        assert data["message"] == "User prompt was compressed"
        assert data["request_id"] == "req-abc-999"
        assert data["event"] == "prompt_compressed"
        assert data["original_tokens"] == 150
        assert data["compressed_tokens"] == 85
        assert "timestamp" in data
    finally:
        current_request_id.reset(token)


def test_text_formatter_with_request_id():
    formatter = TextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    token = current_request_id.set("req-xyz-12345")
    try:
        record = logging.LogRecord(
            name="optillm.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Cache hit occurred",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "[req-xyz-]" in output
        assert "Cache hit occurred" in output
    finally:
        current_request_id.reset(token)


def test_log_event_helper():
    mock_logger = MagicMock()
    log_event(
        mock_logger,
        event="cache_check",
        level=logging.INFO,
        similarity=0.94,
        cache_hit=True,
    )
    mock_logger.log.assert_called_once()
    args, kwargs = mock_logger.log.call_args
    assert args[0] == logging.INFO
    assert "cache_check" in args[1]
    assert kwargs["extra"]["event"] == "cache_check"
    assert kwargs["extra"]["similarity"] == 0.94


# ── 6.2 OpenTelemetry Distributed Tracing Tests ───────────────────────────────

def test_init_tracing_and_span_generation():
    init_tracing(service_name="optillm-test")
    tracer = get_tracer()
    assert tracer is not None

    with trace_span("optillm.test.span", {"test.attr": "value", "count": 42}) as span:
        assert span is not None
        # Nested child span
        with trace_span("optillm.test.child", {"step": 1}) as child:
            assert child is not None


def test_trace_span_records_exception():
    with pytest.raises(ValueError):
        with trace_span("optillm.failing.span", {"provider": "mock"}) as span:
            raise ValueError("Upstream provider failure")


# ── 6.3 Cost Attribution, Waterfall & SLA Analytics Tests ─────────────────────

def test_get_cost_attribution_service():
    with SessionLocal() as db:
        data = get_cost_attribution(db)

        # Check top-level keys
        assert "summary" in data
        assert "cost_by_model" in data
        assert "cost_by_team" in data
        assert "savings_waterfall" in data
        assert "provider_latencies" in data
        assert "sla_compliance" in data

        # Check waterfall math
        waterfall = data["savings_waterfall"]
        assert "gross_spend_usd" in waterfall
        assert "net_spend_usd" in waterfall
        assert "total_savings_usd" in waterfall
        assert "breakdown" in waterfall
        assert "semantic_cache_usd" in waterfall["breakdown"]
        assert "context_compression_usd" in waterfall["breakdown"]
        assert "model_routing_usd" in waterfall["breakdown"]

        # Gross spend must be >= net spend
        assert waterfall["gross_spend_usd"] >= waterfall["net_spend_usd"]


def test_cost_attribution_and_live_dashboard_endpoints():
    client = TestClient(app)

    # 1. Cost attribution endpoint
    res_attr = client.get("/api/v1/analytics/cost-attribution")
    assert res_attr.status_code == 200
    attr_data = res_attr.json()
    assert "savings_waterfall" in attr_data
    assert "cost_by_model" in attr_data

    # 2. Live dashboard endpoint
    res_live = client.get("/api/v1/dashboard/live")
    assert res_live.status_code == 200
    live_data = res_live.json()
    assert "savings_waterfall" in live_data
    assert "sla" in live_data
    assert live_data["sla"]["status"] in ["compliant", "at_risk"]


# ── 6.4 Prometheus Metrics & Alerting YAML Validation ─────────────────────────

def test_prometheus_metrics_generation():
    record_request_metrics(
        method="POST",
        endpoint="/api/v1/chat/completions",
        status_code=200,
        duration_seconds=0.125,
    )
    CIRCUIT_BREAKER_STATE.labels(provider="openai").set(0)
    RATE_LIMIT_EXCEEDED_TOTAL.labels(limit_type="rpm").inc()
    DAILY_BUDGET_UTILIZATION.set(0.45)

    metrics_bytes, content_type = render_metrics()
    metrics_text = metrics_bytes.decode("utf-8")

    assert "optillm_http_requests_total" in metrics_text
    assert "optillm_http_request_duration_seconds" in metrics_text
    assert "optillm_circuit_breaker_state" in metrics_text
    assert "optillm_rate_limit_exceeded_total" in metrics_text
    assert "optillm_daily_budget_utilization_ratio" in metrics_text


def test_prometheus_alerts_yaml_validity():
    alerts_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "deploy",
        "prometheus",
        "alerts.yml",
    )
    assert os.path.isfile(alerts_path), f"alerts.yml not found at {alerts_path}"

    with open(alerts_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "groups" in data
    assert len(data["groups"]) >= 1
    rules = data["groups"][0]["rules"]

    rule_names = {r["alert"] for r in rules}
    expected_rules = {
        "OptiLLMHighErrorRate",
        "OptiLLMCacheHitRateLow",
        "OptiLLMProviderCircuitOpen",
        "OptiLLMBudgetNearLimit",
        "OptiLLMSLALatencyBreach",
        "OptiLLMRateLimitSpike",
    }
    assert expected_rules.issubset(rule_names), f"Missing rules: {expected_rules - rule_names}"


def test_prometheus_server_config_yaml_validity():
    prom_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "deploy",
        "prometheus",
        "prometheus.yml",
    )
    assert os.path.isfile(prom_path), f"prometheus.yml not found at {prom_path}"

    with open(prom_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "global" in data
    assert "rule_files" in data
    assert "scrape_configs" in data
