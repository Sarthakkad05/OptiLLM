"""
Request ID Tracing and Request Logging Middleware.
Attaches a unique Request ID to each incoming request, correlates contextvars,
and emits structured HTTP events with execution latency.
"""

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.logging import current_request_id, log_event

logger = logging.getLogger("optillm.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique X-Request-ID header to requests and responses,
    populates current_request_id contextvar, and emits structured events.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        token = current_request_id.set(request_id)

        client_ip = request.client.host if request.client else "127.0.0.1"
        start_time = time.time()

        log_event(
            logger,
            event="http_request_start",
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            request_id=request_id,
        )

        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000.0
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

            log_event(
                logger,
                event="http_request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(process_time, 2),
                request_id=request_id,
            )
            return response
        except Exception as exc:
            process_time = (time.time() - start_time) * 1000.0
            log_event(
                logger,
                event="http_request_error",
                level=logging.ERROR,
                method=request.method,
                path=request.url.path,
                error=str(exc),
                duration_ms=round(process_time, 2),
                request_id=request_id,
            )
            raise exc
        finally:
            current_request_id.reset(token)


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces MAX_REQUEST_BYTES on incoming requests.
    Rejects requests exceeding the limit with HTTP 413 Payload Too Large.
    """

    def __init__(self, app, max_bytes: int = 1_048_576):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        from fastapi.responses import JSONResponse

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Payload too large. Maximum request size is {self.max_bytes} bytes."
                        },
                    )
            except ValueError:
                pass

        return await call_next(request)
