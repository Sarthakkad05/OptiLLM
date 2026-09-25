"""
Teams Management Endpoints.

POST   /v1/teams              — Create team
GET    /v1/teams              — List teams
GET    /v1/teams/{team_id}    — Get team + spend summary
PATCH  /v1/teams/{team_id}   — Update budget / limits
DELETE /v1/teams/{team_id}   — Delete team
"""

import secrets
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Team, RequestLog
from app.db.session import get_db

router = APIRouter(tags=["Teams"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class CreateTeamRequest(BaseModel):
    name: str
    org_id: Optional[str] = None
    daily_budget_usd: float = 100.0
    monthly_budget_usd: float = 1000.0
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None


class UpdateTeamRequest(BaseModel):
    name: Optional[str] = None
    daily_budget_usd: Optional[float] = None
    monthly_budget_usd: Optional[float] = None
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None


class TeamSpend(BaseModel):
    daily_spent_usd: float
    monthly_spent_usd: float
    daily_budget_usd: float
    monthly_budget_usd: float
    daily_remaining_usd: float
    monthly_remaining_usd: float


class TeamResponse(BaseModel):
    team_id: str
    name: str
    org_id: Optional[str]
    spend: TeamSpend
    rpm_limit: Optional[int]
    tpm_limit: Optional[int]
    created_at: str


class TeamsListResponse(BaseModel):
    teams: List[TeamResponse]
    total: int


def _team_to_response(team: Team) -> TeamResponse:
    return TeamResponse(
        team_id=team.team_id,
        name=team.name,
        org_id=team.org_id,
        spend=TeamSpend(
            daily_spent_usd=team.daily_spent_usd,
            monthly_spent_usd=team.monthly_spent_usd,
            daily_budget_usd=team.daily_budget_usd,
            monthly_budget_usd=team.monthly_budget_usd,
            daily_remaining_usd=max(0.0, team.daily_budget_usd - team.daily_spent_usd),
            monthly_remaining_usd=max(0.0, team.monthly_budget_usd - team.monthly_spent_usd),
        ),
        rpm_limit=team.rpm_limit,
        tpm_limit=team.tpm_limit,
        created_at=str(team.created_at),
    )


def _reset_budgets_if_needed(team: Team) -> Team:
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")
    if team.last_reset_day != today:
        team.daily_spent_usd = 0.0
        team.last_reset_day = today
    if team.last_reset_month != month:
        team.monthly_spent_usd = 0.0
        team.last_reset_month = month
    return team


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/v1/teams",
    response_model=TeamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Team",
)
def create_team(request: CreateTeamRequest, db: Session = Depends(get_db)) -> TeamResponse:
    """Creates a new team with budget and rate limit settings."""
    team_id = f"team-{secrets.token_hex(8)}"
    now = datetime.now(timezone.utc)
    team = Team(
        team_id=team_id,
        name=request.name,
        org_id=request.org_id,
        daily_budget_usd=request.daily_budget_usd,
        monthly_budget_usd=request.monthly_budget_usd,
        daily_spent_usd=0.0,
        monthly_spent_usd=0.0,
        last_reset_day=now.strftime("%Y-%m-%d"),
        last_reset_month=now.strftime("%Y-%m"),
        rpm_limit=request.rpm_limit,
        tpm_limit=request.tpm_limit,
    )
    db.add(team)
    db.commit()
    db.refresh(team)
    return _team_to_response(team)


@router.get(
    "/v1/teams",
    response_model=TeamsListResponse,
    summary="List Teams",
)
def list_teams(db: Session = Depends(get_db)) -> TeamsListResponse:
    """Returns all teams with spend summaries."""
    teams = db.query(Team).order_by(Team.created_at.desc()).all()
    for t in teams:
        _reset_budgets_if_needed(t)
    db.commit()
    return TeamsListResponse(
        teams=[_team_to_response(t) for t in teams],
        total=len(teams),
    )


@router.get(
    "/v1/teams/{team_id}",
    response_model=TeamResponse,
    summary="Get Team",
)
def get_team(team_id: str, db: Session = Depends(get_db)) -> TeamResponse:
    """Returns a team's details and current spend summary."""
    team = db.query(Team).filter(Team.team_id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
    _reset_budgets_if_needed(team)
    db.commit()
    return _team_to_response(team)


@router.patch(
    "/v1/teams/{team_id}",
    response_model=TeamResponse,
    summary="Update Team",
)
def update_team(
    team_id: str, request: UpdateTeamRequest, db: Session = Depends(get_db)
) -> TeamResponse:
    """Updates a team's name, budget, or rate limits."""
    team = db.query(Team).filter(Team.team_id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")

    if request.name is not None:
        team.name = request.name
    if request.daily_budget_usd is not None:
        team.daily_budget_usd = request.daily_budget_usd
    if request.monthly_budget_usd is not None:
        team.monthly_budget_usd = request.monthly_budget_usd
    if request.rpm_limit is not None:
        team.rpm_limit = request.rpm_limit
    if request.tpm_limit is not None:
        team.tpm_limit = request.tpm_limit

    db.commit()
    db.refresh(team)
    return _team_to_response(team)


@router.delete(
    "/v1/teams/{team_id}",
    summary="Delete Team",
)
def delete_team(team_id: str, db: Session = Depends(get_db)):
    """Deletes a team. Keys and users are not deleted but lose team association."""
    team = db.query(Team).filter(Team.team_id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found.")
    db.delete(team)
    db.commit()
    return {"team_id": team_id, "status": "deleted"}
