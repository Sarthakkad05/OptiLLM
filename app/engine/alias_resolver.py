"""
Model Alias Resolver
Resolves friendly model aliases to their real provider model names.

Uses an in-memory LRU-style cache (TTL 60s) to avoid DB round-trips on every request.
Falls back to the original model name if no alias is found.
"""

import logging
import time
from functools import lru_cache
from typing import Dict, Optional, Tuple

logger = logging.getLogger("optillm.engine.alias_resolver")

# Simple time-bounded cache: {(alias, team_id): (target_model, cached_at)}
_alias_cache: Dict[Tuple[str, Optional[str]], Tuple[str, float]] = {}
_CACHE_TTL = 60.0  # seconds


def invalidate_cache():
    """Clears the alias resolution cache. Called when aliases are created/deleted."""
    global _alias_cache
    _alias_cache.clear()
    logger.info("Model alias cache invalidated.")


def _is_stale(cached_at: float) -> bool:
    return (time.time() - cached_at) > _CACHE_TTL


def resolve_model(
    model: str,
    team_id: Optional[str] = None,
    db=None,
) -> str:
    """
    Resolves a model name or alias to the real target model.

    Resolution order:
      1. Team-scoped alias (alias + team_id match)
      2. Global alias (alias + team_id=None)
      3. Original model name (no alias found)

    Args:
        model:   The model name from the request (may be an alias).
        team_id: The requesting team's ID (for team-scoped aliases).
        db:      SQLAlchemy Session. If None, DB lookup is skipped.

    Returns:
        The resolved real model name.
    """
    if db is None:
        return model

    # Check team-scoped alias first
    for lookup_team in [team_id, None]:
        cache_key = (model, lookup_team)
        cached = _alias_cache.get(cache_key)
        if cached and not _is_stale(cached[1]):
            target, _ = cached
            if target != model:
                logger.debug(
                    "Alias resolved: '%s' → '%s' (team=%s, cached)",
                    model, target, lookup_team,
                )
            return target

    # DB lookup
    try:
        from app.db.models import ModelAlias

        # Prefer team-scoped alias
        record = None
        if team_id:
            record = (
                db.query(ModelAlias)
                .filter(ModelAlias.alias == model, ModelAlias.team_id == team_id)
                .first()
            )

        # Fall back to global alias
        if not record:
            record = (
                db.query(ModelAlias)
                .filter(ModelAlias.alias == model, ModelAlias.team_id.is_(None))
                .first()
            )

        if record:
            target = record.target_model
            _alias_cache[(model, record.team_id)] = (target, time.time())
            logger.info(
                "Alias resolved: '%s' → '%s' (team=%s)",
                model, target, record.team_id,
            )
            return target

        # Cache negative result (no alias found)
        _alias_cache[(model, None)] = (model, time.time())
        return model

    except Exception as exc:
        logger.warning("Alias resolution DB error: %s — using original model name.", exc)
        return model
