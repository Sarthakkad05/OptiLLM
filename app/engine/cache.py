"""
Semantic Cache Service
Orchestrates embedding generation, FAISS similarity search, and DB response storage.

Workflow:
  CHECK:   text → embedding → FAISS search → if score >= threshold → fetch from DB
  INSERT:  text → embedding → FAISS add → store response + prompt_text in DB

Startup Sync:
  sync_cache_on_startup(db) rebuilds the FAISS index from DB entries.
  This fixes the FAISS↔DB ID drift bug that caused all cache lookups to miss
  after a server restart.

TTL:
  Cache entries can have an optional expires_at timestamp.
  Expired entries are treated as misses (but not deleted — use /api/v1/cache/clear).

Threshold: 0.95 cosine similarity (very strict — prevents false positives)
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from app.engine.embedding import generate_embedding, generate_embeddings_batch
from app.engine import faiss_store
from app.db.models import CacheEntry

logger = logging.getLogger("optillm.engine.cache")

# Cosine similarity threshold for a cache hit.
# 0.95 means vectors must overlap in 95% of their semantic space.
SIMILARITY_THRESHOLD = 0.95


def _extract_lookup_text(messages: list) -> str:
    """
    Extract the text used for cache lookup.
    We use only the last user message for embedding — this is the semantic intent.
    System prompts are excluded to allow the same question across different contexts.
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return " ".join(m.get("content", "") for m in messages)
    return user_messages[-1].get("content", "")


def sync_cache_on_startup(db: Session) -> None:
    """
    Synchronise the FAISS index with the DB on every startup.

    Problem being solved:
        FAISS persists its index to disk. The DB persists CacheEntry rows.
        Both use sequential integer IDs. After a DB reset, migration, or partial
        failure, the IDs drift — a FAISS hit returns an ID that has no matching
        DB row, causing every cache lookup to fall through to a miss.

    Solution:
        1. Load all CacheEntry rows from DB (including prompt_text).
        2. Batch-embed all prompt texts.
        3. Atomically rebuild a fresh FAISS index from those embeddings.
        4. Save to disk. FAISS and DB are now in sync.

    This runs in ~200ms for 1,000 entries (batch embedding is fast).
    """
    entries: List[CacheEntry] = db.query(CacheEntry).order_by(CacheEntry.faiss_index_id).all()

    if not entries:
        logger.info("Cache sync: DB is empty — fresh FAISS index ready.")
        faiss_store.rebuild_from_entries([])
        return

    # Filter out entries with no prompt_text (legacy rows from before this fix)
    valid = [e for e in entries if e.prompt_text and e.prompt_text.strip()]
    skipped = len(entries) - len(valid)

    if skipped:
        logger.warning(
            "Cache sync: %d entries have no prompt_text (pre-fix legacy rows) — they will be skipped.",
            skipped,
        )

    if not valid:
        logger.warning("Cache sync: no valid entries to re-index. Resetting FAISS.")
        faiss_store.rebuild_from_entries([])
        return

    logger.info("Cache sync: re-embedding %d DB entries to rebuild FAISS index...", len(valid))

    # Batch embed all prompts (much faster than one-by-one)
    texts = [e.prompt_text for e in valid]
    import numpy as np
    vectors = generate_embeddings_batch(texts)  # shape (N, 384)

    # Build (faiss_id, vector_row) pairs
    pairs = [
        (e.faiss_index_id, vectors[i : i + 1].astype(np.float32))
        for i, e in enumerate(valid)
    ]

    rebuilt = faiss_store.rebuild_from_entries(pairs)
    logger.info("Cache sync complete — %d vectors in FAISS, %d in DB.", rebuilt, len(entries))


def check_cache(messages: list, db: Session) -> Optional[Dict[str, Any]]:
    """
    Check if a semantically similar prompt exists in cache.

    Returns:
        Dict with response_text, tokens_input, tokens_output, model if hit.
        None if cache miss (including expired entries).
    """
    lookup_text = _extract_lookup_text(messages)
    if not lookup_text.strip():
        return None

    # Generate normalized embedding
    vector = generate_embedding(lookup_text)

    # Search FAISS for nearest neighbor
    distances, indices = faiss_store.search(vector, k=1)
    score = float(distances[0][0])
    faiss_id = int(indices[0][0])

    logger.info("Cache search | score=%.4f | threshold=%.2f", score, SIMILARITY_THRESHOLD)

    if faiss_id == -1 or score < SIMILARITY_THRESHOLD:
        logger.info("Cache MISS (score=%.4f)", score)
        return None

    # Fetch cached response from DB
    entry = db.query(CacheEntry).filter(CacheEntry.faiss_index_id == faiss_id).first()
    if not entry:
        logger.warning(
            "FAISS hit (id=%d, score=%.4f) but no DB entry found — "
            "index may be stale. Run sync_cache_on_startup to fix.",
            faiss_id, score,
        )
        return None

    # TTL check — treat expired entries as misses
    if entry.expires_at is not None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if entry.expires_at < now:
            logger.info("Cache MISS — entry %d is expired (expired_at=%s)", entry.id, entry.expires_at)
            return None

    logger.info("Cache HIT (score=%.4f | faiss_id=%d | model=%s)", score, faiss_id, entry.model)
    return {
        "response_text": entry.response_text,
        "tokens_input": entry.tokens_input,
        "tokens_output": entry.tokens_output,
        "model": entry.model,
        "similarity_score": score,
    }


def insert_cache(
    messages: list,
    response_text: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    db: Session,
    ttl_seconds: Optional[int] = None,
) -> None:
    """
    Insert a new prompt-response pair into the semantic cache.

    Args:
        ttl_seconds: If set, the cache entry expires after this many seconds.
                     None = never expires.
    """
    lookup_text = _extract_lookup_text(messages)
    if not lookup_text.strip():
        logger.warning("Empty lookup text — skipping cache insertion.")
        return

    # Generate embedding and add to FAISS
    vector = generate_embedding(lookup_text)
    faiss_id = faiss_store.add(vector)

    # Calculate expiry
    expires_at = None
    if ttl_seconds is not None:
        from datetime import timedelta
        expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=ttl_seconds)

    # Persist response + prompt_text to DB (prompt_text enables future FAISS rebuilds)
    entry = CacheEntry(
        faiss_index_id=faiss_id,
        prompt_text=lookup_text[:2000],  # cap at 2000 chars — enough for re-embedding
        response_text=response_text,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        expires_at=expires_at,
    )
    db.add(entry)
    db.commit()

    logger.info(
        "Cache INSERT | faiss_id=%d | model=%s | tokens_in=%d | total_cached=%d%s",
        faiss_id, model, tokens_input, faiss_store.total_vectors(),
        f" | expires_at={expires_at}" if expires_at else "",
    )


def clear_cache(db: Session) -> int:
    """
    Wipe all cache entries from DB and reset the FAISS index.
    Returns the number of entries deleted.
    """
    count = db.query(CacheEntry).count()
    db.query(CacheEntry).delete()
    db.commit()
    faiss_store.reset_index()
    logger.info("Cache cleared — %d entries deleted.", count)
    return count


def get_cache_stats(db: Session) -> Dict[str, Any]:
    """Returns semantic cache health metrics."""
    total_entries = db.query(CacheEntry).count()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expired = db.query(CacheEntry).filter(
        CacheEntry.expires_at.isnot(None),
        CacheEntry.expires_at < now,
    ).count()

    return {
        "total_cached_responses": total_entries,
        "expired_entries": expired,
        "active_entries": total_entries - expired,
        "faiss_vector_count": faiss_store.total_vectors(),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
