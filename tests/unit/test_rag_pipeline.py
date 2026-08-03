"""
Unit tests for app/rag/ modules — chunker, ingestion pipeline, QdrantStore, and HybridRetriever.
"""

from app.rag.chunker import chunk_text
from app.rag.ingestion import get_document_pipeline
from app.rag.qdrant_store import QdrantStore, get_qdrant_store
from app.rag.retrieval import HybridRetriever, get_hybrid_retriever


def test_chunker_fixed_and_recursive():
    text = "Paragraph 1 is about AI optimization.\n\nParagraph 2 discusses model routing and caching strategies."

    rec_chunks = chunk_text(text, strategy="recursive", chunk_size=50, overlap=10)
    assert len(rec_chunks) >= 2
    assert any("optimization" in c["text"] for c in rec_chunks)

    fix_chunks = chunk_text(text, strategy="fixed", chunk_size=40, overlap=10)
    assert len(fix_chunks) >= 2


def test_document_pipeline():
    pipeline = get_document_pipeline()
    doc_text = "# OptiLLM Gateway\nOptiLLM sits between applications and LLM providers."

    processed = pipeline.process_document(
        content=doc_text,
        document_name="architecture.md",
        source_type="markdown",
        chunk_size=100,
    )

    assert processed["document_name"] == "architecture.md"
    assert processed["total_chunks"] >= 1
    assert processed["total_chars"] == len(doc_text)


def test_qdrant_store_and_hybrid_retriever():
    store = get_qdrant_store()
    collection_name = "test_unit_rag"

    chunks = [
        {"text": "Python is a high-level dynamic programming language.", "chunk_index": 0, "document_name": "python.txt"},
        {"text": "FAISS and Qdrant are vector search engines for fast similarity lookups.", "chunk_index": 1, "document_name": "vectors.txt"},
    ]

    inserted = store.add_chunks(collection_name, chunks)
    assert inserted == 2

    retriever = get_hybrid_retriever()
    results = retriever.hybrid_search(
        collection_name=collection_name,
        query_text="vector search Qdrant",
        top_k=2,
    )

    assert len(results) >= 1
    assert "Qdrant" in results[0]["text"] or "FAISS" in results[0]["text"]
    assert results[0]["score"] > 0.0

    # Cleanup
    store.delete_collection(collection_name)
