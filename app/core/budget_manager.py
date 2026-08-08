"""
Budget Manager & Pre-Flight Cost Prediction.
Enforces per-key daily and monthly spend limits to prevent budget overruns.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import KeyBudget
from app.services.cost_estimator import estimate_cost
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.budget_manager")


def _get_current_date_tags() -> Tuple[str, str]:
    """Returns current date formatted as (YYYY-MM-DD, YYYY-MM)."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")


def get_or_create_budget(
    api_key: str,
    db: Session,
    tenant_id: str = "default",
) -> KeyBudget:
    """Fetches or creates a KeyBudget row for the given API key and tenant."""
    budget = (
        db.query(KeyBudget)
        .filter(KeyBudget.api_key == api_key, KeyBudget.tenant_id == tenant_id)
        .first()
    )
    today, month = _get_current_date_tags()

    if not budget:
        budget = KeyBudget(
            api_key=api_key,
            tenant_id=tenant_id,
            daily_budget_usd=10.0,
            monthly_budget_usd=100.0,
            daily_spent_usd=0.0,
            monthly_spent_usd=0.0,
            last_reset_day=today,
            last_reset_month=month,
        )
        db.add(budget)
        db.commit()
        db.refresh(budget)
        return budget

    # Reset daily budget if date changed
    if budget.last_reset_day != today:
        budget.daily_spent_usd = 0.0
        budget.last_reset_day = today

    # Reset monthly budget if month changed
    if budget.last_reset_month != month:
        budget.monthly_spent_usd = 0.0
        budget.last_reset_month = month

    db.commit()
    db.refresh(budget)
    return budget


def check_budget_and_predict(
    api_key: str,
    model: str,
    messages: list,
    db: Session,
    predicted_output_tokens: int = 150,
    tenant_id: str = "default",
) -> Tuple[bool, float, str]:
    """
    Pre-flight check predicting request cost and verifying against daily/monthly caps.
    Returns (is_allowed, predicted_cost_usd, reason).
    """
    budget = get_or_create_budget(api_key=api_key, db=db, tenant_id=tenant_id)

    # 1. Pre-flight cost prediction
    tokens_in = count_tokens_in_messages(messages, model)
    predicted_cost = estimate_cost(
        model=model,
        tokens_input=tokens_in,
        tokens_output=predicted_output_tokens,
    )

    # 2. Check Daily Limit
    if (budget.daily_spent_usd + predicted_cost) > budget.daily_budget_usd:
        reason = (
            f"Daily budget limit exceeded (${budget.daily_spent_usd:.4f} spent + "
            f"${predicted_cost:.4f} predicted > ${budget.daily_budget_usd:.2f} limit)"
        )
        logger.warning("[%s] Budget Blocked: %s", api_key, reason)
        return False, predicted_cost, reason

    # 3. Check Monthly Limit
    if (budget.monthly_spent_usd + predicted_cost) > budget.monthly_budget_usd:
        reason = (
            f"Monthly budget limit exceeded (${budget.monthly_spent_usd:.4f} spent + "
            f"${predicted_cost:.4f} predicted > ${budget.monthly_budget_usd:.2f} limit)"
        )
        logger.warning("[%s] Budget Blocked: %s", api_key, reason)
        return False, predicted_cost, reason

    return True, predicted_cost, "Budget check passed."


def record_spend(
    api_key: str,
    cost_usd: float,
    db: Session,
    tenant_id: str = "default",
) -> KeyBudget:
    """Updates daily & monthly spent amounts for the given API key and tenant."""
    budget = get_or_create_budget(api_key=api_key, db=db, tenant_id=tenant_id)
    budget.daily_spent_usd = round(budget.daily_spent_usd + cost_usd, 6)
    budget.monthly_spent_usd = round(budget.monthly_spent_usd + cost_usd, 6)

    db.commit()
    db.refresh(budget)

    logger.info(
        "[%s] Recorded spend +$%.6f (Daily: $%.4f/$%.2f | Monthly: $%.4f/$%.2f)",
        api_key,
        cost_usd,
        budget.daily_spent_usd,
        budget.daily_budget_usd,
        budget.monthly_spent_usd,
        budget.monthly_budget_usd,
    )
    return budget
