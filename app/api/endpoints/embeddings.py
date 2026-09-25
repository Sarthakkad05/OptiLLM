"""
POST /v1/embeddings — OpenAI-compatible embeddings endpoint.

Routes embedding requests to the appropriate provider (OpenAI, Gemini).
Returns embeddings in the standard OpenAI format:
  {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}], ...}
"""

import logging
import time
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.auth import verify_api_key
from app.core.rate_limiter import check_rate_limit
from app.db.session import get_db

router = APIRouter()
logger = logging.getLogger("optillm.api.embeddings")

# Embedding models and their provider mapping
_EMBEDDING_MODEL_PROVIDER = {
    "text-embedding-3-small": "openai",
    "text-embedding-3-large": "openai",
    "text-embedding-ada-002": "openai",
    "models/text-embedding-004": "gemini",
    "text-embedding-004": "gemini",
}

_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-004": 768,
    "models/text-embedding-004": 768,
}


# ── Schemas ───────────────────────────────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "text-embedding-3-small"
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: EmbeddingUsage


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    tags=["Embeddings"],
    summary="Create Embeddings",
    description=(
        "OpenAI-compatible embeddings endpoint. Routes to OpenAI or Gemini based on "
        "the requested model. Returns embeddings in the standard OpenAI format."
    ),
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def create_embeddings(
    request: EmbeddingRequest,
    db: Session = Depends(get_db),
) -> EmbeddingResponse:
    """Creates embeddings for the provided input text(s)."""
    # Normalize input to list
    texts = [request.input] if isinstance(request.input, str) else request.input

    provider_name = _EMBEDDING_MODEL_PROVIDER.get(request.model, "openai")

    # Estimate token count (rough approximation: 1 token ≈ 4 chars)
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = max(1, total_chars // 4)

    try:
        if provider_name == "openai":
            from app.providers.openai_client import openai_provider
            if not openai_provider.is_available():
                raise HTTPException(
                    status_code=503,
                    detail="OpenAI provider is not configured. Set OPENAI_API_KEY in .env.",
                )
            embeddings = await openai_provider.embed(texts, model=request.model)

        elif provider_name == "gemini":
            # Gemini embedding via REST
            from app.core.config import settings
            import httpx
            if not settings.GEMINI_API_KEY:
                raise HTTPException(
                    status_code=503,
                    detail="Gemini provider is not configured. Set GEMINI_API_KEY in .env.",
                )
            model_name = request.model.replace("models/", "")
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:batchEmbedContents?key={settings.GEMINI_API_KEY}"
            )
            requests_body = {
                "requests": [
                    {"model": f"models/{model_name}", "content": {"parts": [{"text": t}]}}
                    for t in texts
                ]
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=requests_body)
                resp.raise_for_status()
            data = resp.json()
            embeddings = [item["values"] for item in data["embeddings"]]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported embedding model: {request.model}")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Embedding request failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Embedding provider error: {str(exc)}")

    return EmbeddingResponse(
        data=[
            EmbeddingObject(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ],
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        ),
    )
