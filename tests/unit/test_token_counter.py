from app.services.token_counter import count_tokens_in_messages, count_tokens_in_string


def test_count_tokens_in_string():
    text = "Hello, world! How are you doing today?"
    count = count_tokens_in_string(text, model="gpt-4o")
    assert count > 0


def test_count_tokens_in_messages():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    count = count_tokens_in_messages(messages, model="gpt-4o")
    assert count > 0
