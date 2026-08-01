from app.engine.compressor import compress


def test_compress_short_messages():
    messages = [{"role": "user", "content": "Hello world"}]
    compressed_messages, stats = compress(messages, model="gpt-4o")

    assert compressed_messages == messages
    assert stats["was_compressed"] is False
    assert stats["tokens_saved"] == 0


def test_compress_heuristic_cleaning():
    messages = [
        {"role": "user", "content": "Hello   world \n\n\n\n This   is    a   test."}
    ]
    compressed_messages, stats = compress(messages, model="gpt-4o")

    # Whitespace should be trimmed/normalised
    cleaned_content = compressed_messages[0]["content"]
    assert "   " not in cleaned_content
    assert "\n\n\n" not in cleaned_content
