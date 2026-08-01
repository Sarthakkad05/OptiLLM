"""
Gateway Service
Full optimization pipeline:
  1. Semantic Cache Check      — avoid LLM call entirely
  2. Context Compression       — reduce token count
  3. Model Routing             — use cheapest capable model
  4. LLM Provider Call
  5. Cache Insertion
  6. Cost + Savings Calculation (cache + compression + routing)
  7. RequestLog persistence
  8. Return structured response
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.db.models import RequestLog
from app.engine.cache import check_cache, insert_cache
from app.engine.compressor import compress
from app.engine.router import route
from app.providers.dispatcher import call_provider, stream_provider
from app.services.cost_estimator import (
    estimate_cache_savings,
    estimate_compression_savings,
    estimate_cost,
    estimate_routing_savings,
)
from app.services.token_counter import count_tokens_in_messages

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
    Main gateway entrypoint — runs the full optimization pipeline.
    Returns a dict matching the ChatCompletionResponse schema.
    """
    request_start = time.time()
    request_id = uuid.uuid4().hex[:12]

    # Extract user prompt snippet for request logging
    user_messages = [m for m in messages if m.get("role") == "user"]
    prompt_snippet = user_messages[-1].get("content", "")[:200] if user_messages else ""

    # Pre-count original input tokens
    original_tokens_in = count_tokens_in_messages(messages, model)
    logger.info(
        "[%s] Request | model=%s | tokens=%d", request_id, model, original_tokens_in
    )

    # ── Semantic Cache Check ──────────────────────────────────────────────────
    if not bypass_cache:
        cache_result = check_cache(messages, db)
        if cache_result:
            cached_tokens_in = cache_result["tokens_input"]
            cached_tokens_out = cache_result["tokens_output"]
            cached_model = cache_result["model"]
            savings = estimate_cache_savings(
                cached_model, cached_tokens_in, cached_tokens_out
            )
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
                request_id,
                latency_ms,
                savings,
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
                "routing_reason": None,
                "complexity": None,
            }

    # ── Context Compression ───────────────────────────────────────────────────
    # Applied to messages sent to LLM — NOT to messages used for cache lookup.
    compression_stats = {
        "original_tokens": original_tokens_in,
        "compressed_tokens": original_tokens_in,
        "tokens_saved": 0,
        "was_compressed": False,
        "compression_ratio": 0.0,
    }
    messages_to_send = messages

    if not bypass_compression:
        messages_to_send, compression_stats = compress(messages, model=model)
        if compression_stats["was_compressed"]:
            logger.info(
                "[%s] COMPRESSED | %d → %d tokens (%.1f%% reduction)",
                request_id,
                compression_stats["original_tokens"],
                compression_stats["compressed_tokens"],
                compression_stats["compression_ratio"] * 100,
            )

    # ── Model Routing ─────────────────────────────────────────────────────────
    # Run on ORIGINAL messages (uncompressed) for accurate complexity analysis.
    routing_result = {
        "model_used": model,
        "routed": False,
        "complexity": "unknown",
        "routing_reason": "Routing bypassed.",
        "score_breakdown": {},
    }

    if not bypass_routing:
        routing_result = route(messages, requested_model=model)
        if routing_result["routed"]:
            logger.info(
                "[%s] ROUTED | %s → %s | complexity=%s | score=%s",
                request_id,
                model,
                routing_result["model_used"],
                routing_result["complexity"],
                routing_result.get("score", "?"),
            )

    model_to_use = routing_result["model_used"]

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

    # ── Calculate all savings ────────────────────────────────────────────────
    compression_savings = estimate_compression_savings(
        model=model_to_use,
        original_tokens=compression_stats["original_tokens"],
        compressed_tokens=compression_stats["compressed_tokens"],
    )
    routing_savings = (
        estimate_routing_savings(
            original_model=model,
            routed_model=model_to_use,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
        )
        if routing_result["routed"]
        else 0.0
    )

    total_savings = compression_savings + routing_savings

    # ── Insert into semantic cache (always use ORIGINAL uncompressed messages)
    if not bypass_cache:
        insert_cache(
            messages=messages,
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
        routed=routing_result["routed"],
        latency_ms=total_latency_ms,
        prompt_snippet=prompt_snippet,
    )
    db.add(log)
    db.commit()

    logger.info(
        "[%s] Complete | model_used=%s | cost=$%.6f | compression_savings=$%.6f | routing_savings=$%.6f | latency=%dms",
        request_id,
        model_to_use,
        cost,
        compression_savings,
        routing_savings,
        total_latency_ms,
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
        "routed": routing_result["routed"],
        "model_requested": model,
        "routing_reason": routing_result.get("routing_reason"),
        "complexity": getattr(
            routing_result.get("complexity"),
            "value",
            str(routing_result.get("complexity", "")),
        ),
    }


async def process_stream_request(
    messages: List[Dict],
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    bypass_cache: bool,
    bypass_compression: bool,
    bypass_routing: bool,
    db: Session,
):
    """
    Streaming entrypoint — runs optimization pipeline and yields SSE data chunks.
    Format: 'data: {"id": "...", "object": "chat.completion.chunk", ...}\n\n'
    """
    import json

    request_start = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    user_messages = [m for m in messages if m.get("role") == "user"]
    prompt_snippet = user_messages[-1].get("content", "")[:200] if user_messages else ""

    # ── Context Compression ──
    messages_to_send = messages
    compression_stats = {
        "original_tokens": 0,
        "compressed_tokens": 0,
        "tokens_saved": 0,
        "was_compressed": False,
    }
    if not bypass_compression:
        messages_to_send, compression_stats = compress(messages, model=model)

    # ── Model Routing ──
    routing_result = {"model_used": model, "routed": False}
    if not bypass_routing:
        routing_result = route(messages, requested_model=model)

    model_to_use = routing_result["model_used"]
    accumulated_content = []

    # Stream from provider
    async for text_chunk in stream_provider(
        messages=messages_to_send,
        model=model_to_use,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        accumulated_content.append(text_chunk)
        chunk_obj = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_to_use,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text_chunk},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_obj)}\n\n"

    # Final completion chunk with finish_reason='stop'
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_to_use,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    full_text = "".join(accumulated_content)
    total_latency_ms = int((time.time() - request_start) * 1000)

    # Insert into cache & log request
    if not bypass_cache and full_text:
        insert_cache(
            messages=messages,
            response_text=full_text,
            model=model_to_use,
            tokens_input=compression_stats.get("compressed_tokens", 0),
            tokens_output=len(full_text.split()),
            db=db,
        )

    log = RequestLog(
        model_requested=model,
        model_used=model_to_use,
        provider="stream",
        tokens_input=compression_stats.get("original_tokens", 0),
        tokens_output=len(full_text.split()),
        tokens_saved=compression_stats.get("tokens_saved", 0),
        cost_usd=0.0,
        savings_usd=0.0,
        cache_hit=False,
        compressed=compression_stats.get("was_compressed", False),
        routed=routing_result.get("routed", False),
        latency_ms=total_latency_ms,
        prompt_snippet=prompt_snippet,
    )
    db.add(log)
    db.commit()
