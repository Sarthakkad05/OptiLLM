from app.engine.structured_output import (
    get_json_schema_instructions,
    parse_structured_json,
)


def test_parse_structured_json_clean_json():
    raw = '{"name": "OptiLLM", "version": "1.0"}'
    parsed = parse_structured_json(raw)
    assert parsed["name"] == "OptiLLM"
    assert parsed["version"] == "1.0"


def test_parse_structured_json_markdown_block():
    raw = 'Here is the response:\n```json\n{"summary": "Test summary"}\n```\n'
    parsed = parse_structured_json(raw)
    assert parsed["summary"] == "Test summary"


def test_json_schema_instructions_generation():
    instructions = get_json_schema_instructions(
        schema_name="UserSummary",
        properties={"username": "string", "score": "number"},
    )
    assert "UserSummary" in instructions
    assert '"username": "<string>"' in instructions
