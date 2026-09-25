"""
Guardrails API Endpoints.

GET  /api/v1/guardrails        — List all registered guardrail hooks and their status
PATCH /api/v1/guardrails/{name} — Enable or disable a guardrail by name
POST /api/v1/guardrails/test   — Test a prompt against all active guardrails (dry-run)
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.guardrail_manager import get_guardrail_manager

router = APIRouter(tags=["Guardrails"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class GuardrailInfo(BaseModel):
    name: str
    mode: str
    enabled: bool
    block_on_trigger: bool


class GuardrailListResponse(BaseModel):
    guardrails: List[GuardrailInfo]
    total: int


class GuardrailUpdateRequest(BaseModel):
    enabled: bool


class GuardrailUpdateResponse(BaseModel):
    name: str
    enabled: bool
    message: str


class GuardrailTestRequest(BaseModel):
    messages: List[dict]


class HookTestResult(BaseModel):
    hook: str
    enabled: bool
    allowed: bool
    warnings: List[str]
    blocked_reason: Optional[str] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None


class GuardrailTestResponse(BaseModel):
    overall_allowed: bool
    hooks: List[HookTestResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/guardrails",
    response_model=GuardrailListResponse,
    summary="List Guardrails",
    description="Returns all registered guardrail hooks with their current enabled state.",
)
def list_guardrails() -> GuardrailListResponse:
    """Returns all configured guardrail hooks."""
    mgr = get_guardrail_manager()
    hooks = mgr.list_hooks()
    return GuardrailListResponse(
        guardrails=[GuardrailInfo(**h) for h in hooks],
        total=len(hooks),
    )


@router.patch(
    "/guardrails/{name}",
    response_model=GuardrailUpdateResponse,
    summary="Enable/Disable Guardrail",
    description="Enables or disables a guardrail hook by name at runtime (no restart needed).",
)
def update_guardrail(
    name: str,
    body: GuardrailUpdateRequest,
) -> GuardrailUpdateResponse:
    """Enables or disables a named guardrail hook."""
    mgr = get_guardrail_manager()
    found = mgr.set_enabled(name, body.enabled)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Guardrail '{name}' not found. Use GET /api/v1/guardrails to see available hooks.",
        )
    state = "enabled" if body.enabled else "disabled"
    return GuardrailUpdateResponse(
        name=name,
        enabled=body.enabled,
        message=f"Guardrail '{name}' has been {state}.",
    )


@router.post(
    "/guardrails/test",
    response_model=GuardrailTestResponse,
    summary="Test Prompt Against Guardrails",
    description=(
        "Dry-runs a prompt through all registered pre-call guardrail hooks. "
        "Does NOT send any request to an LLM — purely for validation."
    ),
)
def test_guardrails(body: GuardrailTestRequest) -> GuardrailTestResponse:
    """Tests a prompt against all active guardrails without making an LLM call."""
    mgr = get_guardrail_manager()
    result = mgr.test_prompt(body.messages)
    return GuardrailTestResponse(
        overall_allowed=result["overall_allowed"],
        hooks=[HookTestResult(**h) for h in result["hooks"]],
    )
