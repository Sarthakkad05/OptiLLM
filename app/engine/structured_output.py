"""
Structured Output Parser Engine.
Uses LangChain utilities and Pydantic validation to enforce and parse structured
JSON responses from raw LLM text generation.
"""

import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger("optillm.engine.structured_output")


def extract_json_block(text: str) -> str:
    """Extracts JSON block enclosed in ```json ... ``` or raw JSON object."""
    match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match_raw = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
    if match_raw:
        return match_raw.group(1).strip()

    return text.strip()


def parse_structured_json(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from raw LLM completion response.
    Raises ValueError if JSON is malformed or unparseable.
    """
    json_str = extract_json_block(raw_text)
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}
    except Exception as exc:
        logger.error("Failed to parse JSON structured output: %s", exc)
        raise ValueError(f"Invalid JSON response: {exc}")


def get_json_schema_instructions(schema_name: str, properties: Dict[str, str]) -> str:
    """
    Generates structured JSON schema prompt instructions.
    """
    props_formatted = ",\n".join([f'  "{k}": "<{v}>"' for k, v in properties.items()])
    return (
        f"You must respond ONLY with a valid JSON object matching the '{schema_name}' schema.\n"
        f"Do not include any conversational text outside the JSON block.\n\n"
        f"JSON Format:\n{{\n{props_formatted}\n}}"
    )
