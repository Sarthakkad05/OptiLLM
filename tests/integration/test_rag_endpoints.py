"""
Integration tests for Phase 12 RAG API endpoints.
"""

import pytest


@pytest.mark.asyncio
async def test_rag_ingest_and_query_flow(async_client):
    collection_name = "test_integ_collection"

    # 1. Ingest document
    ingest_payload = {
        "collection_name": collection_name,
        "content": "OptiLLM provides semantic cache, context compression, model routing, and RAG support.",
        "document_name": "optillm_summary.txt",
        "source_type": "text",
        "chunk_size": 200,
    }
    ingest_res = await async_client.post("/api/v1/rag/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["collection_name"] == collection_name
    assert ingest_data["total_chunks"] >= 1

    # 2. Query collection
    query_payload = {
        "collection_name": collection_name,
        "query": "context compression and routing",
        "top_k": 3,
    }
    query_res = await async_client.post("/api/v1/rag/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert query_data["total_retrieved"] >= 1
    assert "retrieved_chunks" in query_data

    # 3. List collections
    list_res = await async_client.get("/api/v1/rag/collections")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert any(c["collection_name"] == collection_name for c in list_data["collections"])

    # 4. Delete collection
    del_res = await async_client.delete(f"/api/v1/rag/collections/{collection_name}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"
