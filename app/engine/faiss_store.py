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

import os
import faiss
import numpy as np
import threading
import logging

logger = logging.getLogger("optillm.engine.faiss_store")

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension
FAISS_INDEX_PATH = "faiss_store/index.faiss"

_index: faiss.IndexFlatIP | None = None
_lock = threading.Lock()  # Thread-safe insertions


def _create_index() -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(VECTOR_DIM)


def load_index() -> faiss.IndexFlatIP:
    """Load FAISS index from disk, or create a new one if not found."""
    global _index
    if _index is not None:
        return _index

    os.makedirs("faiss_store", exist_ok=True)

    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading FAISS index from disk: %s", FAISS_INDEX_PATH)
        _index = faiss.read_index(FAISS_INDEX_PATH)
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
    index = get_index()
    os.makedirs("faiss_store", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
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
    """Wipe the index (for testing only)."""
    global _index
    _index = _create_index()
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    logger.warning("FAISS index has been reset.")
