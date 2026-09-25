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
from app.engine.alias_resolver import resolve_model
from app.providers.dispatcher import call_provider, stream_provider
from app.core.guardrail_manager import get_guardrail_manager
from app.core.callback_manager import get_callback_manager
from app.core.tracing import trace_span
from app.services.cost_estimator import (
    estimate_cache_savings,
    estimate_compression_savings,
    estimate_cost,
    estimate_routing_savings,
)
from app.services.token_counter import count_tokens_in_messages, count_tokens_in_string

logger = logging.getLogger("optillm.gateway")


async def process_request(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    bypass_cache: bool = False,
    bypass_compression: bool = False,
    bypass_routing: bool = False,
    bypass_guardrails: bool = False,
    cache_threshold: Optional[float] = None,
    cache_namespace: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    compression_mode: str = "smart",
    db: Session = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main gateway entrypoint — runs the full optimization pipeline.
    Returns a dict matching the ChatCompletionResponse schema.
    """
    import datetime
    request_start = time.time()
    start_time_iso = datetime.datetime.utcnow().isoformat() + "Z"
    request_id = uuid.uuid4().hex[:12]

    # ── Step 0: Model Alias Resolution ────────────────────────────────────
    model = resolve_model(model, team_id=None, db=db)

    # Extract user prompt snippet for request logging
    user_messages = [m for m in messages if m.get("role") == "user"]
    prompt_snippet = user_messages[-1].get("content", "")[:200] if user_messages else ""

    # Pre-count original input tokens
    original_tokens_in = count_tokens_in_messages(messages, model)
    logger.info(
        "[%s] Request | model=%s | tokens=%d | ns=%s",
        request_id,
        model,
        original_tokens_in,
        cache_namespace or "default",
    )

    # ── Plugins: Pre-process Hook ─────────────────────────────────────────────
    from app.plugins.registry import get_plugin_registry
    plugin_req = get_plugin_registry().run_pre_process({
        "messages": messages,
        "model": model,
        "request_id": request_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    messages = plugin_req.get("messages", messages)
    model = plugin_req.get("model", model)

    # ── Pre-call Guardrails ───────────────────────────────────────────────────
    guardrail_warnings: List[str] = []
    if not bypass_guardrails:
        guardrail_mgr = get_guardrail_manager()
        allowed, messages, guardrail_warnings, blocked_reason = guardrail_mgr.run_pre_call(messages)
        if not allowed:
            raise ValueError(f"Guardrail blocked request: {blocked_reason}")
        if guardrail_warnings:
            logger.info("[%s] Guardrail warnings: %s", request_id, guardrail_warnings)

    # ── Semantic Cache Check ──────────────────────────────────────────────────
    if not bypass_cache:
        with trace_span("optillm.cache.check", {
            "namespace": cache_namespace or "default",
            "similarity_threshold": cache_threshold,
        }) as cache_span:
            cache_result = check_cache(
                messages,
                db,
                namespace=cache_namespace,
                similarity_threshold=cache_threshold,
            )
            if cache_span and cache_result:
                cache_span.set_attribute("cache_hit", True)
                cache_span.set_attribute("similarity_score", float(cache_result.get("similarity", 1.0)))
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
                tag=tag,
            )

            db.add(log)
            db.commit()

            logger.info(
                "[%s] CACHE HIT | latency=%dms | saved=$%.6f",
                request_id,
                latency_ms,
                savings,
            )
            return get_plugin_registry().run_post_process({
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
            })

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
        with trace_span("optillm.compress", {
            "compression_mode": compression_mode,
            "original_tokens": original_tokens_in,
        }) as comp_span:
            messages_to_send, compression_stats = compress(
                messages, model=model, mode=compression_mode
            )
            if comp_span:
                comp_span.set_attribute("compressed_tokens", compression_stats["compressed_tokens"])
                comp_span.set_attribute("tokens_saved", compression_stats["tokens_saved"])
        if compression_stats["was_compressed"]:
            logger.info(
                "[%s] COMPRESSED | %d → %d tokens (%.1f%% reduction, mode=%s)",
                request_id,
                compression_stats["original_tokens"],
                compression_stats["compressed_tokens"],
                compression_stats["compression_ratio"] * 100,
                compression_mode,
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
        with trace_span("optillm.route", {"requested_model": model}) as route_span:
            routing_result = route(messages, requested_model=model)
            if route_span:
                route_span.set_attribute("routed_model", routing_result["model_used"])
                route_span.set_attribute("routed", routing_result["routed"])
                route_span.set_attribute("complexity", routing_result.get("complexity", "unknown"))
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

    # Return this request's DB connection to the pool before the (slow) provider
    # await. This handler runs sync DB calls on the event loop, so if in-flight
    # requests held their connections across the await, the 16th concurrent
    # request would block the loop waiting on an exhausted pool — and the
    # holders could never resume to release theirs (found under load test).
    db.commit()

    # ── Call provider ────────────────────────────────────────────────────────
    logger.info("[%s] Calling provider | model=%s", request_id, model_to_use)
    with trace_span("optillm.provider.call", {"model": model_to_use, "stream": False}) as prov_span:
        provider_response = await call_provider(
            messages=messages_to_send,
            model=model_to_use,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if prov_span:
            prov_span.set_attribute("provider", provider_response.get("provider", "unknown"))
            prov_span.set_attribute("tokens_input", provider_response.get("tokens_input", 0))
            prov_span.set_attribute("tokens_output", provider_response.get("tokens_output", 0))

    total_latency_ms = int((time.time() - request_start) * 1000)
    tokens_in = provider_response["tokens_input"]
    tokens_out = provider_response["tokens_output"]
    cost = estimate_cost(model_to_use, tokens_in, tokens_out)

    # ── Post-call Guardrails ─────────────────────────────────────────────────
    if not bypass_guardrails:
        guardrail_mgr = get_guardrail_manager()
        provider_response["content"], post_warnings = guardrail_mgr.run_post_call(
            provider_response["content"], messages
        )
        guardrail_warnings.extend(post_warnings)

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
        with trace_span("optillm.cache.insert", {
            "model": provider_response["model"],
            "namespace": cache_namespace or "default",
            "ttl_seconds": ttl_seconds,
        }):
            insert_cache(
                messages=messages,
                response_text=provider_response["content"],
                model=provider_response["model"],
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                db=db,
                namespace=cache_namespace,
                ttl_seconds=ttl_seconds,
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
        shadow_disagreement=routing_result.get("shadow_disagreement", False),
        ai_predicted_complexity=routing_result.get("ai_predicted_complexity"),
        latency_ms=total_latency_ms,
        prompt_snippet=prompt_snippet,
        tag=tag,
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

    result = {
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
        "guardrail_warnings": guardrail_warnings,
    }

    # ── Fire Observability Callbacks (non-blocking) ────────────────────────────
    import datetime
    end_time_iso = datetime.datetime.utcnow().isoformat() + "Z"
    callback_payload = {
        **result,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "start_time_iso": start_time_iso,
        "end_time_iso": end_time_iso,
    }
    get_callback_manager().fire_success(callback_payload)

    result = get_plugin_registry().run_post_process(result)
    return result


async def process_stream_request(
    messages: List[Dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    bypass_cache: bool = False,
    bypass_compression: bool = False,
    bypass_routing: bool = False,
    bypass_guardrails: bool = False,
    db: Session = None,
    tag: Optional[str] = None,
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

    # ── Pre-call Guardrails (same as non-streaming path) ──────────────────────
    if not bypass_guardrails:
        guardrail_mgr = get_guardrail_manager()
        allowed, messages, guardrail_warnings, blocked_reason = guardrail_mgr.run_pre_call(messages)
        if not allowed:
            import json
            error_chunk = json.dumps({
                "error": {"message": f"Guardrail blocked request: {blocked_reason}", "type": "guardrail_error"}
            })
            yield f"data: {error_chunk}\n\n"
            yield "data: [DONE]\n\n"
            return
        if guardrail_warnings:
            logger.info("[%s] Stream guardrail warnings: %s", request_id, guardrail_warnings)

    # ── Context Compression ──
    messages_to_send = messages
    compression_stats = {
        "original_tokens": 0,
        "compressed_tokens": 0,
        "tokens_saved": 0,
        "was_compressed": False,
    }
    if not bypass_compression:
        with trace_span("optillm.compress", {"compression_mode": "smart", "stream": True}):
            messages_to_send, compression_stats = compress(messages, model=model)

    # ── Model Routing ──
    routing_result = {"model_used": model, "routed": False}
    if not bypass_routing:
        with trace_span("optillm.route", {"requested_model": model, "stream": True}):
            routing_result = route(messages, requested_model=model)

    model_to_use = routing_result["model_used"]
    accumulated_content = []

    # Release the DB connection before streaming — see process_request.
    db.commit()

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

    # Accurate token counts using tiktoken — NOT word-count approximation
    tokens_in = compression_stats.get("compressed_tokens", 0)
    tokens_out = count_tokens_in_string(full_text, model_to_use)
    stream_cost = estimate_cost(model_to_use, tokens_in, tokens_out)

    # Routing savings for stream path
    stream_routing_savings = (
        estimate_routing_savings(
            original_model=model,
            routed_model=model_to_use,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
        )
        if routing_result.get("routed", False)
        else 0.0
    )
    stream_compression_savings = estimate_compression_savings(
        model=model_to_use,
        original_tokens=compression_stats.get("original_tokens", 0),
        compressed_tokens=tokens_in,
    )
    stream_total_savings = stream_routing_savings + stream_compression_savings

    # Insert into cache & log request
    if not bypass_cache and full_text:
        insert_cache(
            messages=messages,
            response_text=full_text,
            model=model_to_use,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            db=db,
        )

    log = RequestLog(
        model_requested=model,
        model_used=model_to_use,
        provider="stream",
        tokens_input=compression_stats.get("original_tokens", 0),
        tokens_output=tokens_out,
        tokens_saved=compression_stats.get("tokens_saved", 0),
        cost_usd=stream_cost,
        savings_usd=stream_total_savings,
        cache_hit=False,
        compressed=compression_stats.get("was_compressed", False),
        routed=routing_result.get("routed", False),
        shadow_disagreement=routing_result.get("shadow_disagreement", False),
        ai_predicted_complexity=routing_result.get("ai_predicted_complexity"),
        latency_ms=total_latency_ms,
        prompt_snippet=prompt_snippet,
        tag=tag,
    )

    db.add(log)
    db.commit()
