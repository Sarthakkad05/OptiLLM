"""
Request ID Tracing and Request Logging Middleware.
Attaches a unique Request ID to each incoming request and logs request duration.
"""

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger("optillm.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique X-Request-ID header to requests and responses
    and logs request method, URL path, status code, and execution time.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.time()
        logger.info(
            "[%s] HTTP %s %s",
            request_id[:8],
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000.0
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"

            logger.info(
                "[%s] HTTP %s %s -> %d (%.2fms)",
                request_id[:8],
                request.method,
                request.url.path,
                response.status_code,
                process_time,
            )
            return response
        except Exception as exc:
            process_time = (time.time() - start_time) * 1000.0
            logger.error(
                "[%s] HTTP %s %s -> Exception (%.2fms): %s",
                request_id[:8],
                request.method,
                request.url.path,
                process_time,
                str(exc),
            )
            raise exc
