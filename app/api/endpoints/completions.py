"""
POST /v1/completions — OpenAI-compatible legacy text completion endpoint.

Accepts a plain-text prompt (not messages array) and routes it through
the full OptiLLM optimization pipeline by wrapping it into a chat message.
Required for compatibility with older OpenAI SDK versions and non-chat clients.
"""

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.auth import verify_api_key
from app.core.budget_manager import check_budget_and_predict, record_spend
from app.core.config import settings
from app.core.rate_limiter import check_rate_limit, record_token_usage
from app.db.session import get_db, release_db_on_return
from app.services.gateway import process_request

router = APIRouter()
logger = logging.getLogger("optillm.api.completions")


# ── Schemas ───────────────────────────────────────────────────────────────────


class CompletionRequest(BaseModel):
    model: str = "gpt-4o"
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    suffix: Optional[str] = None
    n: Optional[int] = 1
    stop: Optional[List[str]] = None
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"
    logprobs: None = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/v1/completions",
    response_model=CompletionResponse,
    tags=["Gateway"],
    summary="Legacy Text Completions",
    description=(
        "OpenAI-compatible legacy text completion endpoint. "
        "Wraps the prompt into a chat message and routes through the OptiLLM pipeline. "
        "Use /v1/chat/completions for new integrations."
    ),
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)],
)
@release_db_on_return
async def text_completions(
    request: CompletionRequest,
    db: Session = Depends(get_db),
    authorization: Optional[str] = Header(None),
) -> CompletionResponse:
    """
    Legacy text completion endpoint. Internally wraps prompt as a chat user message
    and passes it through the full OptiLLM optimization pipeline.
    """
    import time

    api_key = "default-key"
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "").strip()

    if len(request.prompt) > settings.MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Prompt too long ({len(request.prompt)} chars). Maximum allowed is {settings.MAX_PROMPT_LENGTH} characters.",
        )

    # Wrap plain prompt as a chat message
    messages = [{"role": "user", "content": request.prompt}]

    # Budget check
    allowed, predicted_cost, reason = check_budget_and_predict(
        api_key=api_key,
        model=request.model,
        messages=messages,
        db=db,
    )
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)

    try:
        result = await process_request(
            messages=messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=request.max_tokens,
            db=db,
        )
        record_spend(api_key=api_key, cost_usd=result.get("cost_usd", 0.0), db=db)
        tokens_used = result.get("tokens_input", 0) + result.get("tokens_output", 0)
        record_token_usage(api_key, tokens_used)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Completions error: %s", e, exc_info=True)
        raise HTTPException(status_code=502, detail=f"Provider error: {str(e)}")

    return CompletionResponse(
        id=result.get("id", f"cmpl-{uuid.uuid4().hex[:8]}"),
        created=result.get("created", int(time.time())),
        model=result.get("model", request.model),
        choices=[
            CompletionChoice(
                text=result["content"],
                index=0,
                finish_reason=result.get("finish_reason", "stop"),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=result.get("tokens_input", 0),
            completion_tokens=result.get("tokens_output", 0),
            total_tokens=result.get("tokens_input", 0) + result.get("tokens_output", 0),
        ),
    )
