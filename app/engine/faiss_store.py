"""
FAISS Vector Store
Manages the in-memory FAISS index for semantic similarity search.
Persists the index to disk so it survives restarts.

Index type: IndexFlatIP (Inner Product)
  - Exact nearest-neighbor search (no approximation)
  - Works as cosine similarity when vectors are L2-normalized
  - Appropriate for MVP scale (< 100k entries)

Persistence:
  - Index is saved to faiss_store/index.faiss on every insertion
  - Loaded from disk on startup (if exists)
"""

import logging
import os
import threading

import faiss
import numpy as np

from app.core.config import settings

logger = logging.getLogger("optillm.engine.faiss_store")

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension

_index: faiss.IndexFlatIP | None = None
_lock = threading.Lock()  # Thread-safe insertions


def _faiss_path() -> str:
    """Returns the configured FAISS index file path."""
    return settings.FAISS_INDEX_PATH


def _create_index() -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(VECTOR_DIM)


def load_index() -> faiss.IndexFlatIP:
    """Load FAISS index from disk, or create a new one if not found."""
    global _index
    if _index is not None:
        return _index

    path = _faiss_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        logger.info("Loading FAISS index from disk: %s", path)
        _index = faiss.read_index(path)
        logger.info("FAISS index loaded — %d vectors.", _index.ntotal)
    else:
        logger.info("No existing FAISS index found. Creating fresh index.")
        _index = _create_index()

    return _index


def get_index() -> faiss.IndexFlatIP:
    if _index is None:
        return load_index()
    return _index


def save_index() -> None:
    """Persist the current FAISS index to disk."""
    path = _faiss_path()
    index = get_index()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path)
    logger.debug("FAISS index saved to disk — %d vectors.", index.ntotal)


def search(vector: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors.

    Args:
        vector: Normalized float32 array of shape (1, VECTOR_DIM)
        k: Number of results to return

    Returns:
        (distances, indices) — distances are cosine similarities [0, 1]
        Returns ([-1], [-1]) if index is empty.
    """
    index = get_index()

    if index.ntotal == 0:
        return np.array([[-1.0]]), np.array([[-1]])

    distances, indices = index.search(vector, k)
    return distances, indices


def add(vector: np.ndarray) -> int:
    """
    Add a normalized vector to the index.

    Returns:
        The FAISS index ID of the newly added vector (0-based).
    """
    with _lock:
        index = get_index()
        index.add(vector)
        new_id = index.ntotal - 1
        save_index()
        logger.debug("Vector added to FAISS index. New total: %d", index.ntotal)
        return new_id


def total_vectors() -> int:
    """Returns total number of vectors in the index."""
    return get_index().ntotal


def reset_index() -> None:
    """Wipe the in-memory index and delete the persisted file."""
    global _index
    path = _faiss_path()
    _index = _create_index()
    if os.path.exists(path):
        os.remove(path)
    logger.warning("FAISS index has been reset.")


def rebuild_from_entries(entries: list) -> int:
    """
    Atomically rebuild the FAISS index from a list of (faiss_id, vector) pairs.
    Called on startup to re-sync FAISS with the DB, fixing the ID-drift bug.

    Args:
        entries: List of (faiss_index_id: int, vector: np.ndarray shape (1, 384))
                 Must be sorted by faiss_index_id ascending.

    Returns:
        Number of vectors re-indexed.
    """
    global _index
    if not entries:
        logger.info("rebuild_from_entries: no DB entries — keeping fresh index.")
        _index = _create_index()
        save_index()
        return 0

    new_index = _create_index()
    # FAISS IndexFlatIP assigns IDs sequentially. We must insert in order.
    sorted_entries = sorted(entries, key=lambda e: e[0])
    expected_id = 0
    vectors_added = 0

    for faiss_id, vector in sorted_entries:
        # If there's a gap (deleted entry), pad with a zero vector so IDs stay consistent
        while expected_id < faiss_id:
            pad = np.zeros((1, VECTOR_DIM), dtype=np.float32)
            new_index.add(pad)
            expected_id += 1
        new_index.add(vector)
        expected_id += 1
        vectors_added += 1

    with _lock:
        _index = new_index
        save_index()

    logger.info(
        "FAISS index rebuilt from DB — %d vectors re-indexed (total=%d).",
        vectors_added,
        new_index.ntotal,
    )
    return vectors_added
