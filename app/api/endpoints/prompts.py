"""
GET /api/v1/prompts
POST /api/v1/prompts/register
POST /api/v1/prompts/render
POST /api/v1/prompts/parse
Prompt template versioning, variable injection, and structured output parsing.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.engine.prompt_templates import prompt_engine
from app.engine.structured_output import (
    get_json_schema_instructions,
    parse_structured_json,
)

router = APIRouter()


class RegisterPromptRequest(BaseModel):
    name: str
    version: str = "v1"
    system_template: str
    user_template: str
    description: Optional[str] = None


class RenderPromptRequest(BaseModel):
    name: str
    version: str = "v1"
    variables: Dict[str, Any]


class ParseStructuredRequest(BaseModel):
    text: str


class SchemaInstructionRequest(BaseModel):
    schema_name: str
    properties: Dict[str, str]


@router.get("", tags=["Prompts"])
def list_prompts() -> List[Dict[str, Any]]:
    """List all registered prompt templates, versions, and expected input variables."""
    return prompt_engine.list_templates()


@router.post("/register", tags=["Prompts"])
def register_prompt(request: RegisterPromptRequest) -> Dict[str, Any]:
    """Register or update a versioned prompt template."""
    record = prompt_engine.register(
        name=request.name,
        version=request.version,
        system_template=request.system_template,
        user_template=request.user_template,
        description=request.description,
    )
    return {
        "name": record.name,
        "version": record.version,
        "description": record.description,
        "input_variables": record.input_variables,
        "message": f"Prompt template '{record.name}:{record.version}' registered successfully.",
    }


@router.post("/render", tags=["Prompts"])
def render_prompt(request: RenderPromptRequest) -> Dict[str, Any]:
    """
    Renders a registered prompt template with variables into standard OpenAI messages list.
    """
    try:
        messages = prompt_engine.render(
            name=request.name,
            version=request.version,
            variables=request.variables,
        )
        return {
            "name": request.name,
            "version": request.version,
            "messages": messages,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Rendering error: {exc}")


@router.post("/parse", tags=["Prompts"])
def parse_structured_output(request: ParseStructuredRequest) -> Dict[str, Any]:
    """
    Extracts and parses JSON from raw LLM completion response.
    """
    try:
        data = parse_structured_json(request.text)
        return {"success": True, "data": data}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/schema_instructions", tags=["Prompts"])
def get_schema_instructions(request: SchemaInstructionRequest) -> Dict[str, Any]:
    """
    Returns formatting instructions to inject into system prompt for structured output.
    """
    instructions = get_json_schema_instructions(
        schema_name=request.schema_name, properties=request.properties
    )
    return {"instructions": instructions}


# ── Version Control Endpoints (Phase 2.5) ──────────────────────────────────────


class CreateVersionRequest(BaseModel):
    system_template: str
    user_template: str
    description: Optional[str] = None
    commit_message: Optional[str] = None
    version: Optional[str] = None


class CompareVersionsRequest(BaseModel):
    version_a: str = "v1"
    version_b: str = "v2"
    variables: Dict[str, Any] = {}


@router.get("/{name}/versions", tags=["Prompts"])
def get_prompt_versions(name: str) -> List[Dict[str, Any]]:
    """List full version history and changelog for a specific prompt template."""
    history = prompt_engine.get_history(name)
    if not history:
        raise HTTPException(status_code=404, detail=f"Prompt template '{name}' not found.")
    return history


@router.post("/{name}/versions", tags=["Prompts"])
def create_prompt_version(name: str, request: CreateVersionRequest) -> Dict[str, Any]:
    """Create a new version for a prompt template (auto-increments if not specified)."""
    record = prompt_engine.create_version(
        name=name,
        system_template=request.system_template,
        user_template=request.user_template,
        description=request.description,
        commit_message=request.commit_message,
        version=request.version,
    )
    return {
        "name": record.name,
        "version": record.version,
        "commit_message": record.commit_message,
        "description": record.description,
        "input_variables": record.input_variables,
        "is_active": record.is_active,
        "message": f"Created version '{record.name}:{record.version}' successfully.",
    }


@router.post("/{name}/rollback/{version}", tags=["Prompts"])
def rollback_prompt_version(name: str, version: str) -> Dict[str, Any]:
    """Roll back the active version of a prompt template to a specified earlier version."""
    try:
        record = prompt_engine.rollback(name, version)
        return {
            "name": record.name,
            "active_version": record.version,
            "description": record.description,
            "message": f"Rolled back '{name}' to active version '{version}'.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/{name}/compare", tags=["Prompts"])
def compare_prompt_versions(name: str, request: CompareVersionsRequest) -> Dict[str, Any]:
    """Compare two versions of a prompt template with sample variables, returning messages and token diff."""
    try:
        return prompt_engine.compare_versions(
            name=name,
            version_a=request.version_a,
            version_b=request.version_b,
            variables=request.variables,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
