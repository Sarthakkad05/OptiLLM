"""
Users Management Endpoints.

POST   /v1/users              — Create user
GET    /v1/users              — List users (filterable by team)
GET    /v1/users/{user_id}    — Get user details + spend
PATCH  /v1/users/{user_id}   — Update user role / budget
DELETE /v1/users/{user_id}   — Delete user
"""

import secrets
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.session import get_db

router = APIRouter(tags=["Users"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class CreateUserRequest(BaseModel):
    email: Optional[str] = None
    team_id: Optional[str] = None
    role: str = "developer"  # admin | developer | read_only
    daily_budget_usd: Optional[float] = None


class UpdateUserRequest(BaseModel):
    email: Optional[str] = None
    team_id: Optional[str] = None
    role: Optional[str] = None
    daily_budget_usd: Optional[float] = None


class UserResponse(BaseModel):
    user_id: str
    email: Optional[str]
    team_id: Optional[str]
    role: str
    daily_budget_usd: Optional[float]
    daily_spent_usd: float
    daily_remaining_usd: Optional[float]
    created_at: str


class UsersListResponse(BaseModel):
    users: List[UserResponse]
    total: int


def _user_to_response(user: User) -> UserResponse:
    remaining = None
    if user.daily_budget_usd is not None:
        remaining = max(0.0, user.daily_budget_usd - user.daily_spent_usd)
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        team_id=user.team_id,
        role=user.role,
        daily_budget_usd=user.daily_budget_usd,
        daily_spent_usd=user.daily_spent_usd,
        daily_remaining_usd=remaining,
        created_at=str(user.created_at),
    )


def _reset_user_budget_if_needed(user: User) -> User:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if user.last_reset_day != today:
        user.daily_spent_usd = 0.0
        user.last_reset_day = today
    return user


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/v1/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create User",
)
def create_user(request: CreateUserRequest, db: Session = Depends(get_db)) -> UserResponse:
    """Creates a new user with optional team association and budget."""
    user_id = f"user-{secrets.token_hex(8)}"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    user = User(
        user_id=user_id,
        email=request.email,
        team_id=request.team_id,
        role=request.role,
        daily_budget_usd=request.daily_budget_usd,
        daily_spent_usd=0.0,
        last_reset_day=today,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _user_to_response(user)


@router.get(
    "/v1/users",
    response_model=UsersListResponse,
    summary="List Users",
)
def list_users(
    team_id: Optional[str] = Query(None, description="Filter users by team ID"),
    db: Session = Depends(get_db),
) -> UsersListResponse:
    """Lists all users, optionally filtered by team."""
    query = db.query(User)
    if team_id:
        query = query.filter(User.team_id == team_id)
    users = query.order_by(User.created_at.desc()).all()
    for u in users:
        _reset_user_budget_if_needed(u)
    db.commit()
    return UsersListResponse(
        users=[_user_to_response(u) for u in users],
        total=len(users),
    )


@router.get(
    "/v1/users/{user_id}",
    response_model=UserResponse,
    summary="Get User",
)
def get_user(user_id: str, db: Session = Depends(get_db)) -> UserResponse:
    """Returns a user's details and current spend."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    _reset_user_budget_if_needed(user)
    db.commit()
    return _user_to_response(user)


@router.patch(
    "/v1/users/{user_id}",
    response_model=UserResponse,
    summary="Update User",
)
def update_user(
    user_id: str, request: UpdateUserRequest, db: Session = Depends(get_db)
) -> UserResponse:
    """Updates a user's role, team, or budget."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    if request.email is not None:
        user.email = request.email
    if request.team_id is not None:
        user.team_id = request.team_id
    if request.role is not None:
        valid_roles = {"admin", "developer", "read_only"}
        if request.role not in valid_roles:
            raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {valid_roles}")
        user.role = request.role
    if request.daily_budget_usd is not None:
        user.daily_budget_usd = request.daily_budget_usd

    db.commit()
    db.refresh(user)
    return _user_to_response(user)


@router.delete(
    "/v1/users/{user_id}",
    summary="Delete User",
)
def delete_user(user_id: str, db: Session = Depends(get_db)):
    """Deletes a user record."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    db.delete(user)
    db.commit()
    return {"user_id": user_id, "status": "deleted"}
