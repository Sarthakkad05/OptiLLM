"""
RFC 7807 Problem Details Error Handler.
Converts HTTPExceptions and unhandled runtime exceptions into RFC 7807 standard JSON responses.
"""

import logging
from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("optillm.errors")

STATUS_TITLE_MAP = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def build_problem_details(
    status_code: int, detail: Any, instance_path: str
) -> Dict[str, Any]:
    """Generates an RFC 7807 problem details dictionary."""
    title = STATUS_TITLE_MAP.get(status_code, "HTTP Error")
    type_slug = title.lower().replace(" ", "-")

    return {
        "type": f"https://optillm.ai/errors/{type_slug}",
        "title": title,
        "status": status_code,
        "detail": str(detail),
        "instance": instance_path,
    }


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for FastAPI HTTPException converting to RFC 7807 format."""
    problem = build_problem_details(
        status_code=exc.status_code,
        detail=exc.detail,
        instance_path=request.url.path,
    )
    headers = getattr(exc, "headers", None) or {}
    return JSONResponse(
        status_code=exc.status_code,
        content=problem,
        headers={"Content-Type": "application/problem+json", **headers},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unexpected internal server errors converting to RFC 7807 format."""
    logger.error("Unhandled exception: %s", str(exc), exc_info=True)
    problem = build_problem_details(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An internal server error occurred.",
        instance_path=request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=problem,
        headers={"Content-Type": "application/problem+json"},
    )
