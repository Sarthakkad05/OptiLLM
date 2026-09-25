"""
GET /v1/models — OpenAI-compatible model listing endpoint.
Returns available models with metadata including context window, pricing, and capabilities.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.services.cost_estimator import get_model_pricing
from app.providers.registry import provider_registry

router = APIRouter()


class ModelCapabilities(BaseModel):
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_structured_output: bool = False
    context_window: int = 4096


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str
    context_window: Optional[int] = None
    input_cost_per_1m: Optional[float] = None
    output_cost_per_1m: Optional[float] = None
    capabilities: Optional[ModelCapabilities] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# Full model registry with metadata
MODEL_REGISTRY = {
    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o": {
        "owned_by": "openai",
        "context_window": 128000,
        "supports_tools": True,
        "supports_structured_output": True,
    },
    "gpt-4o-mini": {
        "owned_by": "openai",
        "context_window": 128000,
        "supports_tools": True,
        "supports_structured_output": True,
    },
    "gpt-4-turbo": {
        "owned_by": "openai",
        "context_window": 128000,
        "supports_tools": True,
        "supports_structured_output": False,
    },
    "gpt-4": {
        "owned_by": "openai",
        "context_window": 8192,
        "supports_tools": True,
        "supports_structured_output": False,
    },
    "gpt-3.5-turbo": {
        "owned_by": "openai",
        "context_window": 16385,
        "supports_tools": True,
        "supports_structured_output": False,
    },
    # ── Google Gemini ──────────────────────────────────────────────────────────
    "gemini-2.0-flash": {
        "owned_by": "google",
        "context_window": 1048576,
        "supports_tools": True,
        "supports_structured_output": True,
    },
    "gemini-1.5-pro": {
        "owned_by": "google",
        "context_window": 2097152,
        "supports_tools": True,
        "supports_structured_output": True,
    },
    "gemini-1.5-flash": {
        "owned_by": "google",
        "context_window": 1048576,
        "supports_tools": True,
        "supports_structured_output": True,
    },
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-3-5-sonnet-20241022": {
        "owned_by": "anthropic",
        "context_window": 200000,
        "supports_tools": True,
        "supports_structured_output": False,
    },
    "claude-3-5-haiku-20241022": {
        "owned_by": "anthropic",
        "context_window": 200000,
        "supports_tools": True,
        "supports_structured_output": False,
    },
    "claude-3-opus-20240229": {
        "owned_by": "anthropic",
        "context_window": 200000,
        "supports_tools": True,
        "supports_structured_output": False,
    },
    # ── Ollama (Local) ────────────────────────────────────────────────────────
    "llama3": {
        "owned_by": "meta-ollama",
        "context_window": 8192,
        "supports_tools": False,
        "supports_structured_output": False,
    },
    "llama3.1": {
        "owned_by": "meta-ollama",
        "context_window": 131072,
        "supports_tools": False,
        "supports_structured_output": False,
    },
    "mistral": {
        "owned_by": "mistral-ollama",
        "context_window": 32768,
        "supports_tools": False,
        "supports_structured_output": False,
    },
}


@router.get(
    "/v1/models",
    response_model=ModelsResponse,
    tags=["Models"],
    summary="List Available Models",
    description=(
        "Returns models available through OptiLLM with pricing and capability metadata. "
        "Compatible with the OpenAI GET /v1/models endpoint."
    ),
)
def list_models() -> ModelsResponse:
    """Returns all models registered in OptiLLM's model registry."""
    available_providers = set(provider_registry.list_available_providers())

    data = []
    for model_id, meta in MODEL_REGISTRY.items():
        owner = meta["owned_by"]
        # Determine provider from owned_by
        provider_key = owner.split("-")[0]  # "openai", "google", "anthropic", "meta"
        provider_map = {
            "openai": "openai",
            "google": "gemini",
            "anthropic": "anthropic",
            "meta": "ollama",
            "mistral": "ollama",
        }
        mapped_provider = provider_map.get(provider_key, "openai")

        pricing = get_model_pricing(model_id)

        data.append(
            ModelCard(
                id=model_id,
                owned_by=owner,
                context_window=meta.get("context_window"),
                input_cost_per_1m=pricing[0],
                output_cost_per_1m=pricing[1],
                capabilities=ModelCapabilities(
                    supports_streaming=True,
                    supports_tools=meta.get("supports_tools", False),
                    supports_structured_output=meta.get("supports_structured_output", False),
                    context_window=meta.get("context_window", 4096),
                ),
            )
        )

    return ModelsResponse(data=data)


@router.get(
    "/v1/models/{model_id}",
    response_model=ModelCard,
    tags=["Models"],
    summary="Get Model Details",
)
def get_model(model_id: str) -> ModelCard:
    """Returns details for a specific model."""
    from fastapi import HTTPException
    meta = MODEL_REGISTRY.get(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

    pricing = get_model_pricing(model_id)
    return ModelCard(
        id=model_id,
        owned_by=meta["owned_by"],
        context_window=meta.get("context_window"),
        input_cost_per_1m=pricing[0],
        output_cost_per_1m=pricing[1],
        capabilities=ModelCapabilities(
            supports_streaming=True,
            supports_tools=meta.get("supports_tools", False),
            supports_structured_output=meta.get("supports_structured_output", False),
            context_window=meta.get("context_window", 4096),
        ),
    )
