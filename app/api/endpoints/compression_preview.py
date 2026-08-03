"""
POST /api/v1/compressor/preview
Endpoint allowing developers to preview context compression on prompt messages,
view side-by-side prompt diffs, and inspect token savings before running completions.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.engine.compressor import compress

router = APIRouter()


class CompressionPreviewRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = "gpt-4o"
    mode: Optional[str] = "smart"
    max_tokens: Optional[int] = 2000


class CompressionPreviewResponse(BaseModel):
    original_messages: List[Dict[str, Any]]
    compressed_messages: List[Dict[str, Any]]
    stats: Dict[str, Any]


@router.post(
    "/compressor/preview",
    response_model=CompressionPreviewResponse,
    tags=["Optimization"],
    summary="Preview Prompt Compression Diff",
    description="Returns compressed prompt messages side-by-side with original messages and token savings metrics.",
)
def preview_compression(request: CompressionPreviewRequest):
    compressed, stats = compress(
        messages=request.messages,
        model=request.model or "gpt-4o",
        max_tokens=request.max_tokens or 2000,
        mode=request.mode or "smart",
    )
    return CompressionPreviewResponse(
        original_messages=request.messages,
        compressed_messages=compressed,
        stats=stats,
    )
