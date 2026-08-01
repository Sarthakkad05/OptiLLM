from app.core.budget_manager import (
    check_budget_and_predict,
    get_or_create_budget,
    record_spend,
)


def test_budget_manager_creation_and_record_spend(db_session):
    api_key = "test_key_123"
    budget = get_or_create_budget(api_key, db_session)

    assert budget.api_key == api_key
    assert budget.daily_spent_usd == 0.0

    record_spend(api_key, cost_usd=0.05, db=db_session)
    updated = get_or_create_budget(api_key, db_session)
    assert updated.daily_spent_usd == 0.05
    assert updated.monthly_spent_usd == 0.05


def test_budget_manager_preflight_check(db_session):
    api_key = "test_key_preflight"
    budget = get_or_create_budget(api_key, db_session)
    budget.daily_budget_usd = 0.0001  # Set extremely low limit
    db_session.commit()

    messages = [{"role": "user", "content": "Hello, cost prediction check"}]
    allowed, predicted, reason = check_budget_and_predict(
        api_key, "gpt-4o", messages, db_session
    )

    assert allowed is False
    assert "Daily budget limit exceeded" in reason
