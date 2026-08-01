"""
GET /api/v1/budgets
POST /api/v1/budgets
Budget management API endpoints for viewing and setting spend limits.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.budget_manager import get_or_create_budget
from app.db.models import KeyBudget
from app.db.session import get_db

router = APIRouter()


class BudgetConfigUpdate(BaseModel):
    api_key: str
    daily_budget_usd: Optional[float] = None
    monthly_budget_usd: Optional[float] = None


class BudgetOut(BaseModel):
    api_key: str
    daily_budget_usd: float
    monthly_budget_usd: float
    daily_spent_usd: float
    monthly_spent_usd: float
    daily_remaining_usd: float
    monthly_remaining_usd: float
    last_reset_day: Optional[str] = None
    last_reset_month: Optional[str] = None


@router.get("", response_model=List[BudgetOut], tags=["Budgets"])
def list_budgets(db: Session = Depends(get_db)):
    """List all API key budgets and remaining allowances."""
    rows = db.query(KeyBudget).all()
    res = []
    for r in rows:
        b = get_or_create_budget(r.api_key, db)
        res.append(
            BudgetOut(
                api_key=b.api_key,
                daily_budget_usd=b.daily_budget_usd,
                monthly_budget_usd=b.monthly_budget_usd,
                daily_spent_usd=b.daily_spent_usd,
                monthly_spent_usd=b.monthly_spent_usd,
                daily_remaining_usd=max(
                    0.0, round(b.daily_budget_usd - b.daily_spent_usd, 4)
                ),
                monthly_remaining_usd=max(
                    0.0, round(b.monthly_budget_usd - b.monthly_spent_usd, 4)
                ),
                last_reset_day=b.last_reset_day,
                last_reset_month=b.last_reset_month,
            )
        )
    return res


@router.get("/{api_key}", response_model=BudgetOut, tags=["Budgets"])
def get_budget_by_key(api_key: str, db: Session = Depends(get_db)):
    """Fetch budget details and remaining allowance for a single API key."""
    b = get_or_create_budget(api_key, db)
    return BudgetOut(
        api_key=b.api_key,
        daily_budget_usd=b.daily_budget_usd,
        monthly_budget_usd=b.monthly_budget_usd,
        daily_spent_usd=b.daily_spent_usd,
        monthly_spent_usd=b.monthly_spent_usd,
        daily_remaining_usd=max(0.0, round(b.daily_budget_usd - b.daily_spent_usd, 4)),
        monthly_remaining_usd=max(
            0.0, round(b.monthly_budget_usd - b.monthly_spent_usd, 4)
        ),
        last_reset_day=b.last_reset_day,
        last_reset_month=b.last_reset_month,
    )


@router.post("", response_model=BudgetOut, tags=["Budgets"])
def set_budget(update: BudgetConfigUpdate, db: Session = Depends(get_db)):
    """
    Set or update daily & monthly USD caps for an API key.
    """
    b = get_or_create_budget(update.api_key, db)

    if update.daily_budget_usd is not None:
        if update.daily_budget_usd < 0:
            raise HTTPException(status_code=400, detail="daily_budget_usd must be >= 0")
        b.daily_budget_usd = update.daily_budget_usd

    if update.monthly_budget_usd is not None:
        if update.monthly_budget_usd < 0:
            raise HTTPException(
                status_code=400, detail="monthly_budget_usd must be >= 0"
            )
        b.monthly_budget_usd = update.monthly_budget_usd

    db.commit()
    db.refresh(b)

    return BudgetOut(
        api_key=b.api_key,
        daily_budget_usd=b.daily_budget_usd,
        monthly_budget_usd=b.monthly_budget_usd,
        daily_spent_usd=b.daily_spent_usd,
        monthly_spent_usd=b.monthly_spent_usd,
        daily_remaining_usd=max(0.0, round(b.daily_budget_usd - b.daily_spent_usd, 4)),
        monthly_remaining_usd=max(
            0.0, round(b.monthly_budget_usd - b.monthly_spent_usd, 4)
        ),
        last_reset_day=b.last_reset_day,
        last_reset_month=b.last_reset_month,
    )
