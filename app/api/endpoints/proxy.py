"""
POST /v1/chat/completions
OpenAI-compatible proxy endpoint.
Clients can point their OpenAI SDK base_url to this endpoint with zero changes.
"""

import logging
from typing import Optional, Union

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.auth import verify_api_key
from app.core.budget_manager import check_budget_and_predict, record_spend
from app.core.rate_limiter import check_rate_limit
from app.db.session import get_db
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    OptiLLMMetadata,
    UsageInfo,
)
from app.services.gateway import process_request, process_stream_request

router = APIRouter()
logger = logging.getLogger("optillm.proxy")


@router.post(
    "/v1/chat/completions",
    response_model=None,
    tags=["Gateway"],
    summary="OpenAI-Compatible Chat Completions",
    description=(
        "Drop-in replacement for OpenAI's /v1/chat/completions endpoint. "
        "Adds semantic caching, context compression, model routing, and cost analytics."
    ),
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
async def chat_completions(
    request: ChatCompletionRequest,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
    x_optillm_tag: Optional[str] = Header(None, alias="x-optillm-tag"),
) -> Union[ChatCompletionResponse, StreamingResponse]:
    """
    Accepts an OpenAI-style chat completion request, runs it through
    the OptiLLM optimization pipeline, and returns a standard response or SSE stream.
    """
    api_key = "default-key"
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "").strip()

    messages = [m.model_dump() for m in request.messages]
    config = request.optillm or {}

    bypass_cache = getattr(config, "bypass_cache", False)
    bypass_compression = getattr(config, "bypass_compression", False)
    bypass_routing = getattr(config, "bypass_routing", False)

    # Pre-flight Budget Check
    allowed, predicted_cost, reason = check_budget_and_predict(
        api_key=api_key,
        model=request.model,
        messages=messages,
        db=db,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    if request.stream:
        generator = process_stream_request(
            messages=messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            bypass_cache=bypass_cache,
            bypass_compression=bypass_compression,
            bypass_routing=bypass_routing,
            db=db,
            tag=x_optillm_tag,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    try:
        result = await process_request(
            messages=messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            bypass_cache=bypass_cache,
            bypass_compression=bypass_compression,
            bypass_routing=bypass_routing,
            db=db,
            tag=x_optillm_tag,
        )
        # Record actual spend in budget manager
        record_spend(api_key=api_key, cost_usd=result.get("cost_usd", 0.0), db=db)
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
            routing_reason=result.get("routing_reason"),
            complexity=result.get("complexity"),
        ),
    )
