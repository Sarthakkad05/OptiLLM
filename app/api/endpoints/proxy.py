"""
POST /v1/chat/completions
OpenAI-compatible proxy endpoint.
Clients can point their OpenAI SDK base_url to this endpoint with zero changes.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    UsageInfo,
    OptiLLMMetadata,
)
from app.services.gateway import process_request

router = APIRouter()
logger = logging.getLogger("optillm.proxy")


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["Gateway"],
    summary="OpenAI-Compatible Chat Completions",
    description=(
        "Drop-in replacement for OpenAI's /v1/chat/completions endpoint. "
        "Adds semantic caching, context compression, model routing, and cost analytics."
    ),
)
async def chat_completions(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
):
    """
    Accepts an OpenAI-style chat completion request, runs it through
    the OptiLLM optimization pipeline, and returns a standard response.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported in this MVP version of OptiLLM.",
        )

    messages = [m.model_dump() for m in request.messages]
    config = request.optillm or {}

    try:
        result = await process_request(
            messages=messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            bypass_cache=getattr(config, "bypass_cache", False),
            bypass_compression=getattr(config, "bypass_compression", False),
            bypass_routing=getattr(config, "bypass_routing", False),
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Gateway error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=502, detail=f"Provider error: {str(e)}")

    return ChatCompletionResponse(
        id=result["id"],
        created=result["created"],
        model=result["model"],
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=result["content"]),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result["tokens_input"],
            completion_tokens=result["tokens_output"],
            total_tokens=result["tokens_input"] + result["tokens_output"],
        ),
        optillm_metadata=OptiLLMMetadata(
            cache_hit=result["cache_hit"],
            compressed=result["compressed"],
            routed=result["routed"],
            model_requested=result["model_requested"],
            model_used=result["model"],
            latency_ms=result["latency_ms"],
            cost_usd=result["cost_usd"],
            savings_usd=result["savings_usd"],
            tokens_saved=result["tokens_saved"],
        ),
    )
