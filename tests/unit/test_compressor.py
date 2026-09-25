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


def test_compress_tfidf_importance_ranking():
    # Long text with multiple distinct sentences that exceeds small threshold
    long_content = (
        "Quantum computing harnesses the phenomena of quantum mechanics to deliver huge leaps forward. "
        "Ordinary computers use bits that are either zero or one. "
        "Quantum computers use qubits that can exist in multidimensional states. "
        "Superposition and entanglement are the two core quantum principles. "
        "Shor's algorithm can factor integers in polynomial time. "
        "Quantum key distribution provides provably secure cryptographic communication. "
        "In conclusion, quantum computing revolutionizes data processing."
    )
    messages = [
        {"role": "system", "content": "You are a physics expert."},
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": "What aspect of quantum physics interest you?"},
        {"role": "user", "content": "Tell me more about Shor's algorithm and cryptography."},
    ]

    compressed_messages, stats = compress(
        messages, model="gpt-4o", max_tokens=60, mode="smart"
    )

    assert stats["was_compressed"] is True
    assert stats["tokens_saved"] > 0
    assert stats["compressed_tokens"] < stats["original_tokens"]
    # Last message should always be preserved intact
    assert compressed_messages[-1]["content"] == "Tell me more about Shor's algorithm and cryptography."
    # System message role should be preserved
    assert compressed_messages[0]["role"] == "system"
