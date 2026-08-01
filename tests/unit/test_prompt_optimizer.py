from app.engine.prompt_optimizer import deduplicate_messages, summarize_conversation


def test_deduplicate_messages():
    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Hello!"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    deduped, count = deduplicate_messages(messages)
    assert count == 1
    assert len(deduped) == 3
    assert deduped[1]["content"] == "Hello!"


def test_summarize_conversation_under_threshold():
    messages = [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Turn 2"},
    ]

    summarized, was_summarized = summarize_conversation(messages, max_turns=8)
    assert was_summarized is False
    assert len(summarized) == 3


def test_summarize_conversation_exceeding_threshold():
    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"User turn {i}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i}"})

    summarized, was_summarized = summarize_conversation(messages, max_turns=8)
    assert was_summarized is True
    assert len(summarized) < len(messages)
    assert any("[CONVERSATION SUMMARY" in m.get("content", "") for m in summarized)
