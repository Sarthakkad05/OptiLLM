"""
RAG API Endpoints — Ingestion, Hybrid Search, RAG Completions, & Collections
Exposes REST endpoints for document loading, hybrid retrieval, auto-augmented chat completions, and vector collections.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.rag.ingestion import get_document_pipeline
from app.rag.qdrant_store import get_qdrant_store
from app.rag.retrieval import get_hybrid_retriever
from app.schemas.rag import (
    CollectionListResponse,
    IngestDocumentRequest,
    IngestDocumentResponse,
    RAGChunk,
    RAGCollectionSummary,
    RAGCompletionRequest,
    RAGQueryRequest,
    RAGQueryResponse,
)
from app.services.gateway import process_request

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=IngestDocumentResponse, summary="Ingest document into RAG collection")
def ingest_document(payload: IngestDocumentRequest) -> IngestDocumentResponse:
    """
    Parses document content, splits into chunks, extracts embeddings,
    and indexes chunks into specified RAG collection.
    """
    if not payload.content or not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Document content cannot be empty."
        )

    pipeline = get_document_pipeline()
    processed = pipeline.process_document(
        content=payload.content,
        document_name=payload.document_name,
        source_type=payload.source_type,
        strategy=payload.strategy,
        chunk_size=payload.chunk_size,
        overlap=payload.overlap,
    )

    store = get_qdrant_store()
    inserted_count = store.add_chunks(
        collection_name=payload.collection_name,
        chunks=processed["chunks"],
    )

    return IngestDocumentResponse(
        collection_name=payload.collection_name,
        document_name=payload.document_name,
        total_chars=processed["total_chars"],
        total_chunks=inserted_count,
        status="success",
    )


@router.post("/query", response_model=RAGQueryResponse, summary="Hybrid search retrieval")
def query_rag(payload: RAGQueryRequest) -> RAGQueryResponse:
    """
    Perform hybrid retrieval (dense vector similarity + BM25 keyword matching)
    against specified RAG collection.
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Search query string cannot be empty."
        )

    retriever = get_hybrid_retriever()
    results = retriever.hybrid_search(
        collection_name=payload.collection_name,
        query_text=payload.query,
        top_k=payload.top_k,
        alpha=payload.alpha,
    )

    chunks = [
        RAGChunk(
            text=r["text"],
            chunk_index=r["chunk_index"],
            document_name=r["document_name"],
            score=r["score"],
        )
        for r in results
    ]

    return RAGQueryResponse(
        collection_name=payload.collection_name,
        query=payload.query,
        retrieved_chunks=chunks,
        total_retrieved=len(chunks),
    )


@router.post("/completions", summary="RAG-augmented chat completion")
async def rag_completion(
    payload: RAGCompletionRequest, db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    RAG-augmented chat completion endpoint.
    Retrieves top relevant chunks from collection and auto-injects them into context before calling gateway pipeline.
    """
    if not payload.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Messages list cannot be empty."
        )

    user_msgs = [m for m in payload.messages if m.get("role") == "user"]
    query_text = user_msgs[-1].get("content", "") if user_msgs else ""

    retriever = get_hybrid_retriever()
    chunks = retriever.hybrid_search(
        collection_name=payload.collection_name,
        query_text=query_text,
        top_k=payload.top_k,
    )

    context_str = "\n---\n".join([f"[{c['document_name']}]: {c['text']}" for c in chunks])

    augmented_messages = list(payload.messages)
    if context_str:
        rag_system_msg = {
            "role": "system",
            "content": f"Relevant Reference Context:\n{context_str}\n\nUse the above reference context where applicable.",
        }
        augmented_messages.insert(0, rag_system_msg)

    response = await process_request(
        messages=augmented_messages,
        model=payload.model,
        temperature=payload.temperature,
        db=db,
    )

    response["rag_metadata"] = {
        "collection_name": payload.collection_name,
        "retrieved_chunks_count": len(chunks),
        "chunks_used": [
            {"document_name": c["document_name"], "score": c["score"]} for c in chunks
        ],
    }

    return response


@router.get("/collections", response_model=CollectionListResponse, summary="List RAG collections")
def list_collections() -> CollectionListResponse:
    """List all active vector collections and chunk counts."""
    store = get_qdrant_store()
    raw = store.list_collections()
    collections = [RAGCollectionSummary(**item) for item in raw]
    return CollectionListResponse(collections=collections)


@router.delete("/collections/{collection_name}", summary="Delete RAG collection")
def delete_collection(collection_name: str) -> Dict[str, Any]:
    """Delete RAG collection and purge indexed chunks."""
    store = get_qdrant_store()
    store.delete_collection(collection_name)
    return {
        "collection_name": collection_name,
        "status": "deleted",
        "message": f"Collection '{collection_name}' successfully removed.",
    }
