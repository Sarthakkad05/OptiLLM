"""
Semantic Cache Service
Orchestrates embedding generation, FAISS similarity search, and DB response storage.

Workflow:
  CHECK:   text → embedding → FAISS search → if score >= threshold → fetch from DB
  INSERT:  text → embedding → FAISS add → store response in DB

Threshold: 0.95 cosine similarity (very strict — prevents false positives)
"""

import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.engine.embedding import generate_embedding
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


def check_cache(messages: list, db: Session) -> Optional[Dict[str, Any]]:
    """
    Check if a semantically similar prompt exists in cache.

    Returns:
        Dict with response_text, tokens_input, tokens_output, model if hit.
        None if cache miss.
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
        logger.warning("FAISS hit (id=%d) but no DB entry found. Treating as miss.", faiss_id)
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
) -> None:
    """
    Insert a new prompt-response pair into the semantic cache.
    """
    lookup_text = _extract_lookup_text(messages)
    if not lookup_text.strip():
        logger.warning("Empty lookup text — skipping cache insertion.")
        return

    # Generate embedding and add to FAISS
    vector = generate_embedding(lookup_text)
    faiss_id = faiss_store.add(vector)

    # Persist response to DB
    entry = CacheEntry(
        faiss_index_id=faiss_id,
        response_text=response_text,
        model=model,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
    )
    db.add(entry)
    db.commit()

    logger.info(
        "Cache INSERT | faiss_id=%d | model=%s | tokens_in=%d | total_cached=%d",
        faiss_id, model, tokens_input, faiss_store.total_vectors(),
    )


def get_cache_stats(db: Session) -> Dict[str, Any]:
    """Returns cache statistics for the dashboard."""
    total_entries = db.query(CacheEntry).count()
    return {
        "total_cached_responses": total_entries,
        "faiss_vector_count": faiss_store.total_vectors(),
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }
