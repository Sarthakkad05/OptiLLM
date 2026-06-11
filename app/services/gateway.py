"""
Gateway Service — Phase 4 (Context Compression)
Orchestrates the full request lifecycle:
  1. Check semantic cache (FAISS + DB)
  2. Count input tokens (local, no API call)
  3. Apply context compression                ← NEW Phase 4
  4. Call the LLM provider (on cache miss)
  5. Insert response into cache
  6. Calculate cost + savings (cache + compression)  ← UPDATED Phase 4
  7. Persist RequestLog to the database
  8. Return structured response

Phase 5 will inject Model Routing between compression and LLM call.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.providers.dispatcher import call_provider
from app.services.token_counter import count_tokens_in_messages
from app.services.cost_estimator import (
    estimate_cost,
    estimate_cache_savings,
    estimate_compression_savings,
)
from app.engine.cache import check_cache, insert_cache
from app.engine.compressor import compress
from app.db.models import RequestLog

logger = logging.getLogger("optillm.gateway")


async def process_request(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    bypass_cache: bool,
    bypass_compression: bool,
    bypass_routing: bool,
    db: Session,
) -> Dict[str, Any]:
    """
    Main gateway entrypoint.
    Returns a dict matching the ChatCompletionResponse schema.
    """
    request_start = time.time()
    request_id = uuid.uuid4().hex[:12]

    # ── Extract user prompt snippet for dashboard display ────────────────────
    user_messages = [m for m in messages if m.get("role") == "user"]
    prompt_snippet = user_messages[-1].get("content", "")[:200] if user_messages else ""

    # ── Pre-count original input tokens ─────────────────────────────────────
    original_tokens_in = count_tokens_in_messages(messages, model)
    logger.info("[%s] Input tokens: %d", request_id, original_tokens_in)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Semantic Cache Check
    # ─────────────────────────────────────────────────────────────────────────
    if not bypass_cache:
        cache_result = check_cache(messages, db)
        if cache_result:
            cached_tokens_in = cache_result["tokens_input"]
            cached_tokens_out = cache_result["tokens_output"]
            cached_model = cache_result["model"]
            savings = estimate_cache_savings(cached_model, cached_tokens_in, cached_tokens_out)
            latency_ms = int((time.time() - request_start) * 1000)

            log = RequestLog(
                model_requested=model,
                model_used=cached_model,
                provider="cache",
                tokens_input=0,
                tokens_output=0,
                tokens_saved=cached_tokens_in + cached_tokens_out,
                cost_usd=0.0,
                savings_usd=savings,
                cache_hit=True,
                compressed=False,
                routed=False,
                latency_ms=latency_ms,
                prompt_snippet=prompt_snippet,
            )
            db.add(log)
            db.commit()

            logger.info(
                "[%s] CACHE HIT | latency=%dms | saved=$%.6f",
                request_id, latency_ms, savings,
            )

            return {
                "id": f"cache-{uuid.uuid4().hex[:8]}",
                "created": int(time.time()),
                "model": cached_model,
                "content": cache_result["response_text"],
                "finish_reason": "stop",
                "tokens_input": 0,
                "tokens_output": 0,
                "latency_ms": latency_ms,
                "cost_usd": 0.0,
                "savings_usd": savings,
                "tokens_saved": cached_tokens_in + cached_tokens_out,
                "cache_hit": True,
                "compressed": False,
                "routed": False,
                "model_requested": model,
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Context Compression
    # Applied to the messages SENT to the LLM — not to the cache lookup.
    # The original `messages` are still used for cache insertion.
    # ─────────────────────────────────────────────────────────────────────────
    compression_stats = {
        "original_tokens": original_tokens_in,
        "compressed_tokens": original_tokens_in,
        "tokens_saved": 0,
        "was_compressed": False,
        "compression_ratio": 0.0,
    }

    messages_to_send = messages  # default: send original

    if not bypass_compression:
        messages_to_send, compression_stats = compress(messages, model=model)
        if compression_stats["was_compressed"]:
            logger.info(
                "[%s] COMPRESSED | %d → %d tokens | saved=%d (%.1f%%)",
                request_id,
                compression_stats["original_tokens"],
                compression_stats["compressed_tokens"],
                compression_stats["tokens_saved"],
                compression_stats["compression_ratio"] * 100,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5 placeholder: Model Router will inject here
    # ─────────────────────────────────────────────────────────────────────────
    model_to_use = model

    # ── Call provider ────────────────────────────────────────────────────────
    logger.info("[%s] Calling provider | model=%s", request_id, model_to_use)
    provider_response = await call_provider(
        messages=messages_to_send,
        model=model_to_use,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    total_latency_ms = int((time.time() - request_start) * 1000)
    tokens_in = provider_response["tokens_input"]
    tokens_out = provider_response["tokens_output"]
    cost = estimate_cost(model_to_use, tokens_in, tokens_out)

    # ── Calculate compression savings ────────────────────────────────────────
    compression_savings = estimate_compression_savings(
        model=model_to_use,
        original_tokens=compression_stats["original_tokens"],
        compressed_tokens=compression_stats["compressed_tokens"],
    )
    total_savings = compression_savings  # Phase 5 will add routing savings

    # ── Insert into semantic cache (using ORIGINAL uncompressed messages) ────
    if not bypass_cache:
        insert_cache(
            messages=messages,           # Original — not compressed version
            response_text=provider_response["content"],
            model=provider_response["model"],
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            db=db,
        )

    # ── Persist RequestLog ───────────────────────────────────────────────────
    log = RequestLog(
        model_requested=model,
        model_used=provider_response["model"],
        provider=provider_response["provider"],
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        tokens_saved=compression_stats["tokens_saved"],
        cost_usd=cost,
        savings_usd=total_savings,
        cache_hit=False,
        compressed=compression_stats["was_compressed"],
        routed=False,                  # Phase 5 will set True
        latency_ms=total_latency_ms,
        prompt_snippet=prompt_snippet,
    )
    db.add(log)
    db.commit()

    logger.info(
        "[%s] Request complete | cost=$%.6f | compression_savings=$%.6f | latency=%dms",
        request_id, cost, compression_savings, total_latency_ms,
    )

    return {
        "id": provider_response["id"],
        "created": int(time.time()),
        "model": provider_response["model"],
        "content": provider_response["content"],
        "finish_reason": provider_response.get("finish_reason", "stop"),
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
        "latency_ms": total_latency_ms,
        "cost_usd": cost,
        "savings_usd": total_savings,
        "tokens_saved": compression_stats["tokens_saved"],
        "cache_hit": False,
        "compressed": compression_stats["was_compressed"],
        "routed": False,
        "model_requested": model,
    }
