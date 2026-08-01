from app.engine.router import Complexity, route


def test_route_simple_prompt():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    result = route(messages, requested_model="gpt-4o")

    assert "model_used" in result
    assert result["complexity"] in [Complexity.LOW, "low", "medium"]


def test_route_complex_prompt():
    messages = [
        {
            "role": "user",
            "content": (
                "Write a Python script to perform async database migration with SQLAlchemy, "
                "handling deadlocks, transactions, and rollback strategies."
            ),
        }
    ]
    result = route(messages, requested_model="gpt-4o")

    assert "model_used" in result
    assert "score_breakdown" in result
