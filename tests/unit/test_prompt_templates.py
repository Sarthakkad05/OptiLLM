from app.engine.prompt_templates import PromptTemplateEngine


def test_prompt_template_engine_default_templates():
    engine = PromptTemplateEngine()
    templates = engine.list_templates()
    assert len(templates) >= 3

    names = [t["name"] for t in templates]
    assert "qa_assistant" in names
    assert "summarizer" in names
    assert "classifier" in names


def test_prompt_template_register_and_render():
    engine = PromptTemplateEngine()
    engine.register(
        name="custom_template",
        version="v2",
        system_template="Role: {role}",
        user_template="Task: {task}",
    )

    rendered = engine.render(
        name="custom_template",
        version="v2",
        variables={"role": "Engineer", "task": "Write python unit tests"},
    )

    assert len(rendered) == 2
    assert rendered[0]["role"] == "system"
    assert "Engineer" in rendered[0]["content"]
    assert rendered[1]["role"] == "user"
    assert "Write python unit tests" in rendered[1]["content"]
