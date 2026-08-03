"""
Pydantic Schemas for Phase 12 RAG Support
Defines request and response objects for document ingestion, retrieval, RAG completions, and collection management.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestDocumentRequest(BaseModel):
    collection_name: str = Field(..., description="Target collection name (e.g. docs)")
    content: str = Field(..., description="Raw text or Markdown document content")
    document_name: str = Field("document.txt", description="Document filename/identifier")
    source_type: str = Field("text", description="Source format: text | markdown | pdf | docx")
    strategy: str = Field("recursive", description="Chunking strategy: recursive | fixed | semantic")
    chunk_size: int = Field(500, ge=50, le=5000, description="Chunk character size")
    overlap: int = Field(50, ge=0, le=1000, description="Overlap between consecutive chunks")


class IngestDocumentResponse(BaseModel):
    collection_name: str
    document_name: str
    total_chars: int
    total_chunks: int
    status: str = "success"


class RAGQueryRequest(BaseModel):
    collection_name: str = Field(..., description="Collection to query")
    query: str = Field(..., description="Search query string")
    top_k: int = Field(5, ge=1, le=50, description="Number of top chunks to retrieve")
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Dense vector vs BM25 weight (0.6 = 60% dense, 40% BM25)")


class RAGChunk(BaseModel):
    text: str
    chunk_index: int
    document_name: str
    score: float


class RAGQueryResponse(BaseModel):
    collection_name: str
    query: str
    retrieved_chunks: List[RAGChunk]
    total_retrieved: int


class RAGCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="Prompt messages")
    collection_name: str = Field(..., description="RAG collection to retrieve context from")
    model: str = Field("gpt-4o", description="Target LLM model")
    top_k: int = Field(3, ge=1, le=10, description="Number of context chunks to inject")
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class RAGCollectionSummary(BaseModel):
    collection_name: str
    chunk_count: int
    status: str = "active"


class CollectionListResponse(BaseModel):
    collections: List[RAGCollectionSummary]
