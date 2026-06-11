"""
Token Counting Service
Uses tiktoken to count tokens locally before sending to any provider.
This avoids a round-trip just to know the token count.
"""

import tiktoken
from typing import List, Dict
import logging

logger = logging.getLogger("optillm.token_counter")

# Map model names to their tiktoken encoding
_ENCODING_MAP = {
    "gpt-4o": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gemini-1.5-pro": "cl100k_base",    # Approximation — Gemini uses SentencePiece
    "gemini-1.5-flash": "cl100k_base",  # Approximation
    "gemini-2.0-flash": "cl100k_base",  # Approximation
}

_DEFAULT_ENCODING = "cl100k_base"


def _get_encoding(model: str) -> tiktoken.Encoding:
    encoding_name = _ENCODING_MAP.get(model, _DEFAULT_ENCODING)
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens_in_string(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in a plain string."""
    enc = _get_encoding(model)
    return len(enc.encode(text))


def count_tokens_in_messages(messages: List[Dict], model: str = "gpt-4o") -> int:
    """
    Count tokens in an OpenAI-style messages list.
    Includes per-message overhead (role + separator tokens).
    """
    enc = _get_encoding(model)
    total = 0

    for message in messages:
        total += 4  # Every message has ~4 overhead tokens (role, separators)
        for key, value in message.items():
            total += len(enc.encode(str(value)))

    total += 2  # Priming tokens for assistant reply
    return total
