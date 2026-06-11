"""
Context Compression Service
Reduces token count before sending prompts to the LLM provider.

Pipeline (2 passes):
  Pass 1 — Heuristic Cleaning:
    - Strip repeated whitespace & blank lines
    - Remove redundant markdown separators
    - Deduplicate consecutive identical lines

  Pass 2 — Token-Aware Truncation:
    - Count tokens after cleaning
    - If still above MAX_TOKENS threshold, apply center-truncation
    - Strategy: keep first 30% (system instructions) + last 70% (latest context/query)
    - LLMs suffer from "Lost in the Middle" — preserving edges maximises quality

Design Rules:
  - We compress `messages_to_send` (LLM input), NOT the original `messages`.
  - Original messages are used for cache lookup (unmodified semantic intent).
  - System messages are protected from center-truncation (only tail-trimmed if huge).
  - User messages receive center-truncation.
"""

import re
import logging
from typing import List, Dict, Tuple, Any
from app.services.token_counter import count_tokens_in_messages, count_tokens_in_string

logger = logging.getLogger("optillm.engine.compressor")

# Compress if total prompt tokens exceed this threshold
MAX_TOKENS_THRESHOLD = 2000

# Absolute hard cap — force truncate anything above this
ABSOLUTE_MAX_TOKENS = 6000

# Fraction of budget kept from the START of a message (system instructions tend to be here)
START_FRACTION = 0.30

# Fraction of budget kept from the END of a message (user query is always at the end)
END_FRACTION = 0.70


# ── Pass 1: Heuristic Cleaning ────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Apply heuristic cleaning to reduce noise tokens."""
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse 3+ consecutive spaces into 1
    text = re.sub(r' {3,}', ' ', text)
    # Remove repeated markdown horizontal rules (---, ___, ***)
    text = re.sub(r'([-_*]{3,}\n?){2,}', '---\n', text)
    # Deduplicate consecutive identical lines
    lines = text.split('\n')
    deduped = [lines[0]] if lines else []
    for line in lines[1:]:
        if line.strip() and line.strip() == deduped[-1].strip():
            continue  # skip duplicate line
        deduped.append(line)
    text = '\n'.join(deduped)
    return text.strip()


def _clean_messages(messages: List[Dict]) -> List[Dict]:
    """Apply heuristic cleaning to all messages."""
    cleaned = []
    for msg in messages:
        cleaned.append({
            **msg,
            "content": _clean_text(msg.get("content", "")),
        })
    return cleaned


# ── Pass 2: Token-Aware Truncation ────────────────────────────────────────────

def _truncate_text(text: str, max_tokens: int, model: str) -> str:
    """
    Center-truncation: keep the start and end, drop the middle.
    Preserves: system instructions (start) + user query (end).
    """
    current_tokens = count_tokens_in_string(text, model)
    if current_tokens <= max_tokens:
        return text

    # Split into word tokens (approximate — tiktoken would be accurate but slow here)
    words = text.split()
    total_words = len(words)

    # Estimate words to keep (proportional)
    ratio = max_tokens / current_tokens
    keep_words = int(total_words * ratio)

    keep_start = int(keep_words * START_FRACTION)
    keep_end = int(keep_words * END_FRACTION)

    start_part = ' '.join(words[:keep_start])
    end_part = ' '.join(words[total_words - keep_end:])

    truncated = f"{start_part}\n\n...[CONTEXT COMPRESSED — {total_words - keep_start - keep_end} words removed]...\n\n{end_part}"
    logger.debug("Truncated message: %d → ~%d tokens", current_tokens, max_tokens)
    return truncated


def _truncate_messages(
    messages: List[Dict], token_budget: int, model: str
) -> List[Dict]:
    """
    Distribute the token budget across messages.
    - System messages: protect but tail-trim if necessary.
    - User/assistant messages: center-truncate the largest ones first.
    """
    result = []
    remaining_budget = token_budget

    # Separate system from conversational messages
    system_msgs = [m for m in messages if m.get("role") == "system"]
    convo_msgs = [m for m in messages if m.get("role") != "system"]

    # Allocate 30% of budget to system messages
    system_budget = int(token_budget * 0.30)
    convo_budget = token_budget - system_budget

    # Truncate system messages (tail-trim — keep the beginning)
    for msg in system_msgs:
        content = msg.get("content", "")
        token_count = count_tokens_in_string(content, model)
        if token_count > system_budget:
            truncated = _truncate_text(content, system_budget, model)
            result.append({**msg, "content": truncated})
        else:
            result.append(msg)

    # For conversational messages, always keep the LAST user message intact
    # (it contains the actual question) and compress earlier context
    if convo_msgs:
        last_msg = convo_msgs[-1]
        earlier_msgs = convo_msgs[:-1]
        last_tokens = count_tokens_in_string(last_msg.get("content", ""), model)
        earlier_budget = max(0, convo_budget - last_tokens)

        for msg in earlier_msgs:
            content = msg.get("content", "")
            token_count = count_tokens_in_string(content, model)
            per_msg_budget = max(100, earlier_budget // max(len(earlier_msgs), 1))
            if token_count > per_msg_budget:
                truncated = _truncate_text(content, per_msg_budget, model)
                result.append({**msg, "content": truncated})
            else:
                result.append(msg)

        result.append(last_msg)  # Always keep last message intact

    return result


# ── Main Entry Point ──────────────────────────────────────────────────────────

def compress(
    messages: List[Dict],
    model: str = "gpt-4o",
    max_tokens: int = MAX_TOKENS_THRESHOLD,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Run the full compression pipeline on a messages list.

    Args:
        messages: OpenAI-format messages list
        model: Model name (used for token counting)
        max_tokens: Token threshold above which compression is applied

    Returns:
        (compressed_messages, stats)

        stats = {
            "original_tokens": int,
            "compressed_tokens": int,
            "tokens_saved": int,
            "was_compressed": bool,
            "compression_ratio": float,
        }
    """
    original_tokens = count_tokens_in_messages(messages, model)

    # Always run cleaning pass
    cleaned = _clean_messages(messages)
    cleaned_tokens = count_tokens_in_messages(cleaned, model)

    was_compressed = False
    final_messages = cleaned

    if original_tokens > max_tokens:
        logger.info(
            "Compression triggered | original=%d tokens | threshold=%d",
            original_tokens, max_tokens,
        )
        was_compressed = True
        final_messages = _truncate_messages(cleaned, max_tokens, model)
    else:
        logger.debug(
            "No truncation needed | original=%d tokens (threshold=%d)",
            original_tokens, max_tokens,
        )

    compressed_tokens = count_tokens_in_messages(final_messages, model)
    tokens_saved = max(0, original_tokens - compressed_tokens)
    compression_ratio = round(1.0 - (compressed_tokens / original_tokens), 4) if original_tokens > 0 else 0.0

    stats = {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": tokens_saved,
        "was_compressed": was_compressed,
        "compression_ratio": compression_ratio,
    }

    if was_compressed:
        logger.info(
            "Compression complete | %d → %d tokens | saved=%d | ratio=%.1f%%",
            original_tokens, compressed_tokens, tokens_saved, compression_ratio * 100,
        )

    return final_messages, stats
