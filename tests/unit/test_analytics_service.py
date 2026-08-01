from app.db.models import RequestLog
from app.services import analytics as analytics_service


def test_latency_percentiles_calculation(db_session):
    # Insert sample request logs with known latencies
    logs = [
        RequestLog(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            provider="openai",
            latency_ms=100,
        ),
        RequestLog(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            provider="openai",
            latency_ms=200,
        ),
        RequestLog(
            model_requested="gemini-2.0-flash",
            model_used="gemini-2.0-flash",
            provider="gemini",
            latency_ms=300,
        ),
    ]
    for log in logs:
        db_session.add(log)
    db_session.commit()

    res = analytics_service.get_latency_percentiles(db_session)
    assert res["total_requests"] == 3
    assert res["p50_ms"] == 200.0
    assert "openai" in res["per_provider"]
    assert "gemini" in res["per_provider"]


def test_token_trends_aggregation(db_session):
    log = RequestLog(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        provider="openai",
        tokens_input=100,
        tokens_output=50,
        tokens_saved=20,
    )
    db_session.add(log)
    db_session.commit()

    trends = analytics_service.get_token_trends(db_session)
    assert len(trends["trends"]) > 0
    point = trends["trends"][0]
    assert point["tokens_input"] == 100
    assert point["tokens_output"] == 50
    assert point["tokens_saved"] == 20


def test_savings_breakdown(db_session):
    logs = [
        RequestLog(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            provider="cache",
            cache_hit=True,
            savings_usd=0.05,
        ),
        RequestLog(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            provider="openai",
            compressed=True,
            savings_usd=0.02,
        ),
        RequestLog(
            model_requested="gpt-4o",
            model_used="gpt-4o-mini",
            provider="openai",
            routed=True,
            savings_usd=0.03,
        ),
    ]
    for log in logs:
        db_session.add(log)
    db_session.commit()

    breakdown = analytics_service.get_savings_breakdown(db_session)
    assert breakdown["cache_savings_usd"] == 0.05
    assert breakdown["compression_savings_usd"] == 0.02
    assert breakdown["routing_savings_usd"] == 0.03
