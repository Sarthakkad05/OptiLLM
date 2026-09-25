"""
Model Aliases Management Endpoints.

POST   /api/v1/models/aliases        — Create model alias
GET    /api/v1/models/aliases        — List all aliases
DELETE /api/v1/models/aliases/{alias} — Remove alias
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import ModelAlias
from app.db.session import get_db

router = APIRouter(tags=["Model Aliases"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class CreateAliasRequest(BaseModel):
    alias: str
    target_model: str
    provider_override: Optional[str] = None
    team_id: Optional[str] = None  # None = global alias


class AliasResponse(BaseModel):
    alias: str
    target_model: str
    provider_override: Optional[str]
    team_id: Optional[str]
    created_at: str


class AliasListResponse(BaseModel):
    aliases: List[AliasResponse]
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/models/aliases",
    response_model=AliasResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Model Alias",
    description="Creates a friendly model alias (e.g. 'gpt-4-production' → 'gpt-4o'). Optional team scoping.",
)
def create_alias(request: CreateAliasRequest, db: Session = Depends(get_db)) -> AliasResponse:
    """Creates a model alias mapping."""
    # Check for duplicate alias (same alias + team_id combo)
    existing = db.query(ModelAlias).filter(
        ModelAlias.alias == request.alias,
        ModelAlias.team_id == request.team_id,
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Alias '{request.alias}' already exists{' for team ' + request.team_id if request.team_id else ' globally'}.",
        )

    alias = ModelAlias(
        alias=request.alias,
        target_model=request.target_model,
        provider_override=request.provider_override,
        team_id=request.team_id,
    )
    db.add(alias)
    db.commit()
    db.refresh(alias)

    # Invalidate in-memory cache
    from app.engine.alias_resolver import invalidate_cache
    invalidate_cache()

    return AliasResponse(
        alias=alias.alias,
        target_model=alias.target_model,
        provider_override=alias.provider_override,
        team_id=alias.team_id,
        created_at=str(alias.created_at),
    )


@router.get(
    "/models/aliases",
    response_model=AliasListResponse,
    summary="List Model Aliases",
)
def list_aliases(db: Session = Depends(get_db)) -> AliasListResponse:
    """Returns all configured model aliases."""
    aliases = db.query(ModelAlias).order_by(ModelAlias.created_at.desc()).all()
    return AliasListResponse(
        aliases=[
            AliasResponse(
                alias=a.alias,
                target_model=a.target_model,
                provider_override=a.provider_override,
                team_id=a.team_id,
                created_at=str(a.created_at),
            )
            for a in aliases
        ],
        total=len(aliases),
    )


@router.delete(
    "/models/aliases/{alias}",
    summary="Delete Model Alias",
)
def delete_alias(alias: str, db: Session = Depends(get_db)):
    """Removes a model alias by its alias name."""
    record = db.query(ModelAlias).filter(ModelAlias.alias == alias).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Alias '{alias}' not found.")
    db.delete(record)
    db.commit()

    from app.engine.alias_resolver import invalidate_cache
    invalidate_cache()

    return {"alias": alias, "status": "deleted"}
