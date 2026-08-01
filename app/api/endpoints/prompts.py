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
