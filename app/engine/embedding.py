"""
Embedding Service
Wraps SentenceTransformer model to generate normalized dense vectors.
Singleton pattern — model is loaded once at startup and reused.

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Fast inference (~5ms/query on CPU)
  - Strong semantic similarity on short texts
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("optillm.engine.embedding")

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Load the embedding model. Call once at startup."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s ...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model


def get_model() -> SentenceTransformer:
    """Get the cached model instance."""
    if _model is None:
        return load_model()
    return _model


def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a normalized L2 unit vector embedding for a text string.
    Normalization is required for cosine similarity via FAISS IndexFlatIP.

    Returns: np.ndarray of shape (1, 384), dtype=float32
    """
    model = get_model()
    embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def generate_embeddings_batch(texts: List[str]) -> np.ndarray:
    """
    Batch embedding generation. More efficient for bulk processing.
    Returns: np.ndarray of shape (N, 384), dtype=float32
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype(np.float32)
