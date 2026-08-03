"""
Document Ingestion Pipeline
Parses, cleans, extracts metadata, and chunks documents for RAG indexing.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from app.rag.chunker import chunk_text

logger = logging.getLogger("optillm.rag.ingestion")


class DocumentPipeline:
    """
    Ingests raw document content, extracts metadata, and splits into chunks.
    """

    def process_document(
        self,
        content: str,
        document_name: str,
        source_type: str = "text",
        strategy: str = "recursive",
        chunk_size: int = 500,
        overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a document string into chunk objects with enriched metadata.

        Returns:
            {
                "document_name": str,
                "source_type": str,
                "total_chars": int,
                "total_chunks": int,
                "ingested_at": int,
                "chunks": List[Dict],
                "metadata": Dict,
            }
        """
        clean_content = content.strip() if content else ""
        if not clean_content:
            return {
                "document_name": document_name,
                "source_type": source_type,
                "total_chars": 0,
                "total_chunks": 0,
                "ingested_at": int(time.time()),
                "chunks": [],
                "metadata": metadata or {},
            }

        chunks = chunk_text(
            clean_content,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        doc_meta = metadata or {}
        doc_meta.update(
            {
                "document_name": document_name,
                "source_type": source_type,
                "chunk_strategy": strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
            }
        )

        for chunk in chunks:
            chunk["document_name"] = document_name
            chunk["source_type"] = source_type

        return {
            "document_name": document_name,
            "source_type": source_type,
            "total_chars": len(clean_content),
            "total_chunks": len(chunks),
            "ingested_at": int(time.time()),
            "chunks": chunks,
            "metadata": doc_meta,
        }


_pipeline_instance: Optional[DocumentPipeline] = None


def get_document_pipeline() -> DocumentPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DocumentPipeline()
    return _pipeline_instance
