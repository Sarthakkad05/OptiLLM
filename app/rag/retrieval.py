"""
Hybrid Retriever — Dense Vector + BM25 Sparse Keyword Search
Combines embedding cosine similarity with BM25 keyword matching using score fusion.
"""

import math
import re
from typing import Any, Dict, List, Optional

from app.rag.qdrant_store import get_qdrant_store

logger = logging_import = __import__("logging").getLogger("optillm.rag.retrieval")


class BM25Scorer:
    """Simple, fast in-memory BM25 scorer for keyword search."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_tokens = [re.findall(r"\w+", doc.lower()) for doc in corpus]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = (sum(self.doc_len) / max(1, len(corpus))) if corpus else 1.0

        # Compute IDF
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

        self.N = len(corpus)

    def score(self, query: str) -> List[float]:
        q_tokens = re.findall(r"\w+", query.lower())
        scores = [0.0] * self.N

        for token in q_tokens:
            if token not in self.df:
                continue
            df_val = self.df[token]
            idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

            for idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(token)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[idx] / self.avg_doc_len))
                    scores[idx] += idf * ((tf * (self.k1 + 1.0)) / denom)

        # Normalize scores to [0, 1]
        max_s = max(scores) if scores else 1.0
        if max_s > 0:
            scores = [s / max_s for s in scores]
        return scores


class HybridRetriever:
    """
    Performs hybrid retrieval combining dense vector similarity and sparse BM25 search.
    """

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Perform score-fused hybrid search.

        Args:
            collection_name: Target RAG collection
            query_text: User search query
            top_k: Number of chunks to retrieve
            alpha: Weight for dense vector search (0.0 to 1.0). (1 - alpha) used for BM25.
        """
        store = get_qdrant_store()
        dense_results = store.search(collection_name, query_text, top_k=top_k * 2)

        stored_items = store.fallback_store.get(collection_name, [])
        if not stored_items:
            return dense_results[:top_k]

        corpus = [item["text"] for item in stored_items]
        bm25 = BM25Scorer(corpus)
        bm25_scores = bm25.score(query_text)

        # Build combined score dictionary
        combined: Dict[str, Dict[str, Any]] = {}

        for idx, item in enumerate(stored_items):
            key = f"{item['document_name']}_{item['chunk_index']}"
            bm_score = bm25_scores[idx]
            combined[key] = {
                "text": item["text"],
                "chunk_index": item["chunk_index"],
                "document_name": item["document_name"],
                "dense_score": 0.0,
                "bm25_score": bm_score,
            }

        for d_res in dense_results:
            key = f"{d_res['document_name']}_{d_res['chunk_index']}"
            if key in combined:
                combined[key]["dense_score"] = d_res["score"]
            else:
                combined[key] = {
                    "text": d_res["text"],
                    "chunk_index": d_res["chunk_index"],
                    "document_name": d_res["document_name"],
                    "dense_score": d_res["score"],
                    "bm25_score": 0.0,
                }

        # Score fusion
        results = []
        for item in combined.values():
            hybrid_score = (alpha * item["dense_score"]) + ((1.0 - alpha) * item["bm25_score"])
            results.append(
                {
                    "text": item["text"],
                    "chunk_index": item["chunk_index"],
                    "document_name": item["document_name"],
                    "score": round(float(hybrid_score), 4),
                    "dense_score": round(float(item["dense_score"]), 4),
                    "bm25_score": round(float(item["bm25_score"]), 4),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


_global_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = HybridRetriever()
    return _global_retriever
