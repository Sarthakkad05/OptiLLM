"""
Qdrant Vector Store Adapter
Manages collection creation, embedding indexing, vector similarity search,
and collection deletion using qdrant-client.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

from app.core.config import settings
from app.engine.embedding import get_model


logger = logging.getLogger("optillm.rag.qdrant_store")


class QdrantStore:
    """
    Adapter for Qdrant Vector DB.
    Supports in-memory, local file, or remote cluster configurations.
    """

    def __init__(self, location: Optional[str] = None):
        self.location = location or settings.QDRANT_URL or ":memory:"
        self.client: Optional[Any] = None
        self.fallback_store: Dict[str, List[Dict[str, Any]]] = {}

        if HAS_QDRANT:
            try:
                if self.location == ":memory:":
                    self.client = QdrantClient(location=":memory:")
                else:
                    self.client = QdrantClient(url=self.location)
                logger.info("Initialized Qdrant client at %s", self.location)
            except Exception as e:
                logger.warning("Could not connect to Qdrant at %s: %s. Using in-memory store.", self.location, e)
                self.client = None

    def _ensure_collection(self, collection_name: str, vector_size: int = 384):
        if self.client:
            try:
                cols = [c.name for c in self.client.get_collections().collections]
                if collection_name not in cols:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    )
            except Exception as e:
                logger.error("Error creating collection %s in Qdrant: %s", collection_name, e)

    def add_chunks(self, collection_name: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and insert chunk dicts into collection.

        Returns:
            Inserted chunk count.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embedder = get_model()

        embeddings = embedder.encode(texts, convert_to_numpy=True)
        vector_dim = embeddings.shape[1]

        if collection_name not in self.fallback_store:
            self.fallback_store[collection_name] = []

        if self.client:
            try:
                self._ensure_collection(collection_name, vector_size=vector_dim)
                points = []
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    p_id = str(uuid.uuid4())
                    payload = {
                        "text": chunk["text"],
                        "chunk_index": chunk.get("chunk_index", idx),
                        "document_name": chunk.get("document_name", "document"),
                        "source_type": chunk.get("source_type", "text"),
                    }
                    points.append(PointStruct(id=p_id, vector=emb.tolist(), payload=payload))

                self.client.upsert(collection_name=collection_name, points=points)
            except Exception as e:
                logger.error("Failed to upsert to Qdrant: %s. Storing in memory.", e)

        # Always append to fallback memory store for hybrid search & fallback
        for chunk, emb in zip(chunks, embeddings):
            self.fallback_store[collection_name].append(
                {
                    "text": chunk["text"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "document_name": chunk.get("document_name", "document"),
                    "source_type": chunk.get("source_type", "text"),
                    "vector": emb,
                }
            )

        return len(chunks)

    def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform dense vector search for query_text.
        """
        if not query_text or not query_text.strip():
            return []

        embedder = get_model()

        query_vec = embedder.encode([query_text], convert_to_numpy=True)[0]

        if self.client and HAS_QDRANT:
            try:
                cols = [c.name for c in self.client.get_collections().collections]
                if collection_name in cols:
                    results = self.client.query_points(
                        collection_name=collection_name,
                        query=query_vec.tolist(),
                        limit=top_k,
                    ).points

                    output = []
                    for pt in results:
                        output.append(
                            {
                                "text": pt.payload.get("text", ""),
                                "chunk_index": pt.payload.get("chunk_index", 0),
                                "document_name": pt.payload.get("document_name", ""),
                                "score": round(float(pt.score), 4),
                            }
                        )
                    return output
            except Exception as e:
                logger.warning("Qdrant search error, using in-memory fallback: %s", e)

        # In-memory cosine similarity fallback search
        stored = self.fallback_store.get(collection_name, [])
        if not stored:
            return []

        scored = []
        q_norm = np.linalg.norm(query_vec) + 1e-9
        for item in stored:
            vec = item["vector"]
            v_norm = np.linalg.norm(vec) + 1e-9
            sim = float(np.dot(query_vec, vec) / (q_norm * v_norm))
            scored.append(
                {
                    "text": item["text"],
                    "chunk_index": item["chunk_index"],
                    "document_name": item["document_name"],
                    "score": round(sim, 4),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all active collections and chunk counts."""
        collections_info = []
        names = set(self.fallback_store.keys())

        if self.client and HAS_QDRANT:
            try:
                q_cols = [c.name for c in self.client.get_collections().collections]
                names.update(q_cols)
            except Exception:
                pass

        for name in sorted(list(names)):
            chunk_count = len(self.fallback_store.get(name, []))
            collections_info.append(
                {
                    "collection_name": name,
                    "chunk_count": chunk_count,
                    "status": "active",
                }
            )

        return collections_info

    def delete_collection(self, collection_name: str) -> bool:
        """Delete collection and purge memory."""
        if collection_name in self.fallback_store:
            del self.fallback_store[collection_name]

        if self.client and HAS_QDRANT:
            try:
                self.client.delete_collection(collection_name=collection_name)
            except Exception:
                pass

        return True


_global_qdrant_store: Optional[QdrantStore] = None


def get_qdrant_store() -> QdrantStore:
    global _global_qdrant_store
    if _global_qdrant_store is None:
        _global_qdrant_store = QdrantStore()
    return _global_qdrant_store
