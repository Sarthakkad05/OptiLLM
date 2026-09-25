"""
Callbacks Management Endpoints.

GET  /api/v1/callbacks       — List registered callbacks and their status
POST /api/v1/callbacks/test  — Send a test event to all callbacks (dry-run)
"""

from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.callback_manager import get_callback_manager

router = APIRouter(tags=["Callbacks"])


class CallbackInfo(BaseModel):
    name: str
    status: str


class CallbacksListResponse(BaseModel):
    callbacks: List[CallbackInfo]
    total: int


class CallbackTestResponse(BaseModel):
    message: str
    callbacks_fired: int


@router.get(
    "/callbacks",
    response_model=CallbacksListResponse,
    summary="List Callbacks",
    description="Returns all registered observability callbacks and their active status.",
)
def list_callbacks() -> CallbacksListResponse:
    """Lists all registered callbacks."""
    mgr = get_callback_manager()
    callbacks = mgr.list_callbacks()
    return CallbacksListResponse(
        callbacks=[CallbackInfo(**c) for c in callbacks],
        total=len(callbacks),
    )


@router.post(
    "/callbacks/test",
    response_model=CallbackTestResponse,
    summary="Test Callbacks",
    description="Sends a synthetic test success event to all registered callbacks to verify configuration.",
)
def test_callbacks() -> CallbackTestResponse:
    """Fires a test event through all registered callbacks."""
    import time
    import datetime

    mgr = get_callback_manager()
    test_payload = {
        "id": "test-event-0000",
        "model": "gpt-4o",
        "model_requested": "gpt-4o",
        "content": "This is a test callback event from OptiLLM.",
        "messages": [{"role": "user", "content": "Test callback"}],
        "tokens_input": 10,
        "tokens_output": 15,
        "latency_ms": 100,
        "cost_usd": 0.0001,
        "savings_usd": 0.0,
        "cache_hit": False,
        "compressed": False,
        "routed": False,
        "temperature": 0.7,
        "max_tokens": None,
        "start_time_iso": datetime.datetime.utcnow().isoformat() + "Z",
        "end_time_iso": datetime.datetime.utcnow().isoformat() + "Z",
    }
    mgr.fire_success(test_payload)
    count = len(mgr.list_callbacks())
    return CallbackTestResponse(
        message=f"Test event fired to {count} callback(s).",
        callbacks_fired=count,
    )
