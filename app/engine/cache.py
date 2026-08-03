"""
Semantic Cache Service.
Orchestrates distributed Redis vector similarity lookup, FAISS fallback, and DB storage.

Workflow:
  CHECK:   text → embedding → Redis lookup → if miss → FAISS search → if miss → LLM
  INSERT:  text → embedding → FAISS add → store in DB → insert into Redis with TTL & Namespace
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import CacheEntry
from app.engine import faiss_store
from app.engine.embedding import generate_embedding, generate_embeddings_batch
from app.engine.redis_cache import (
    clear_redis_cache,
    insert_redis_cache,
    is_redis_available,
    lookup_redis_cache,
)

logger = logging.getLogger("optillm.engine.cache")

SIMILARITY_THRESHOLD = settings.CACHE_SIMILARITY_THRESHOLD


def _extract_lookup_text(messages: list) -> str:
    """Extract semantic intent text (last user message)."""
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return " ".join(m.get("content", "") for m in messages)
    return user_messages[-1].get("content", "")


def sync_cache_on_startup(db: Session) -> None:
    """Synchronises FAISS index and Redis with DB cache entries on startup."""
    entries: List[CacheEntry] = (
        db.query(CacheEntry).order_by(CacheEntry.faiss_index_id).all()
    )

    if not entries:
        logger.info("Cache sync: DB is empty — fresh FAISS index ready.")
        faiss_store.rebuild_from_entries([])
        return

    valid = [e for e in entries if e.prompt_text and e.prompt_text.strip()]
    if not valid:
        faiss_store.rebuild_from_entries([])
        return

    logger.info("Cache sync: re-indexing %d DB entries...", len(valid))
    texts = [e.prompt_text for e in valid]
    import numpy as np

    vectors = generate_embeddings_batch(texts)

    pairs = [
        (e.faiss_index_id, vectors[i : i + 1].astype(np.float32))
        for i, e in enumerate(valid)
    ]

    rebuilt = faiss_store.rebuild_from_entries(pairs)

    # Populates Redis if connected
    if is_redis_available():
        for i, e in enumerate(valid):
            insert_redis_cache(
                entry_id=str(e.faiss_index_id),
                prompt_text=e.prompt_text,
                query_embedding=vectors[i],
                response_text=e.response_text,
                model=e.model,
                tokens_input=e.tokens_input,
                tokens_output=e.tokens_output,
            )

    logger.info("Cache sync complete — %d entries in FAISS/Redis.", rebuilt)


def check_cache(
    messages: list,
    db: Session,
    namespace: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Check if a semantically similar prompt exists in distributed Redis or FAISS cache.
    """
    lookup_text = _extract_lookup_text(messages)
    if not lookup_text.strip():
        return None

    threshold = (
        similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
    )

    # Generate normalized 384d embedding
    vector = generate_embedding(lookup_text)

    # 1. Distributed Redis Cache Lookup
    redis_match = lookup_redis_cache(
        query_embedding=vector, threshold=threshold, namespace=namespace
    )
    if redis_match:
        return {
            "response_text": redis_match["response_text"],
            "tokens_input": redis_match["tokens_input"],
            "tokens_output": redis_match["tokens_output"],
            "model": redis_match["model"],
            "similarity_score": redis_match["similarity"],
            "source": "redis",
        }

    # 2. Local FAISS Fallback Search
    distances, indices = faiss_store.search(vector, k=1)
    score = float(distances[0][0])
    faiss_id = int(indices[0][0])

    if faiss_id == -1 or score < threshold:
        return None

    entry = db.query(CacheEntry).filter(CacheEntry.faiss_index_id == faiss_id).first()
    if not entry:
        return None

    if entry.expires_at is not None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if entry.expires_at < now:
            return None

    logger.info("FAISS CACHE HIT (score=%.4f | faiss_id=%d)", score, faiss_id)
    return {
        "response_text": entry.response_text,
        "tokens_input": entry.tokens_input,
        "tokens_output": entry.tokens_output,
        "model": entry.model,
        "similarity_score": score,
        "source": "faiss",
    }


def insert_cache(
    messages: list,
    response_text: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    db: Session,
    namespace: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> None:
    """
    Inserts prompt-response pair into FAISS, DB, and distributed Redis cache.
    """
    lookup_text = _extract_lookup_text(messages)
    if not lookup_text.strip():
        return

    vector = generate_embedding(lookup_text)
    faiss_id = faiss_store.add(vector)
    entry_uuid = uuid.uuid4().hex[:12]

    ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS
    expires_at = None
    if ttl > 0:
        from datetime import timedelta

        expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            seconds=ttl
        )

    # 1. Insert into DB & FAISS
    entry = CacheEntry(
        faiss_index_id=faiss_id,
        prompt_text=lookup_text[:2000],
        response_text=response_text,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        expires_at=expires_at,
    )
    db.add(entry)
    db.commit()

    # 2. Dual-write to Redis Cache
    insert_redis_cache(
        entry_id=entry_uuid,
        prompt_text=lookup_text[:2000],
        query_embedding=vector,
        response_text=response_text,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        namespace=namespace,
        ttl_seconds=ttl,
    )

    logger.info(
        "Cache INSERT complete | faiss_id=%d | model=%s | ttl=%ds", faiss_id, model, ttl
    )


def warm_cache(db: Session, namespace: Optional[str] = None) -> int:
    """Pre-loads all DB cache entries into Redis and FAISS vector index."""
    sync_cache_on_startup(db)
    return db.query(CacheEntry).count()


def clear_cache(db: Session, namespace: Optional[str] = None) -> int:
    """Wipe all cache entries from DB, FAISS, and Redis."""
    count = db.query(CacheEntry).count()
    db.query(CacheEntry).delete()
    db.commit()
    faiss_store.reset_index()
    clear_redis_cache(namespace=namespace)
    logger.info("Cache cleared — %d entries removed from DB/FAISS/Redis.", count)
    return count


def get_cache_stats(db: Session) -> Dict[str, Any]:
    """Returns semantic cache health metrics across DB, FAISS, and Redis."""
    total_entries = db.query(CacheEntry).count()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expired = (
        db.query(CacheEntry)
        .filter(
            CacheEntry.expires_at.isnot(None),
            CacheEntry.expires_at < now,
        )
        .count()
    )

    return {
        "total_cached_responses": total_entries,
        "expired_entries": expired,
        "active_entries": total_entries - expired,
        "faiss_vector_count": faiss_store.total_vectors(),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "redis_available": is_redis_available(),
        "redis_url": settings.REDIS_URL or "not_configured",
        "cache_namespace": settings.CACHE_NAMESPACE,
        "cache_ttl_seconds": settings.CACHE_TTL_SECONDS,
    }
