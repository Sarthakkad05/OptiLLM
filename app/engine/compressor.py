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

import logging
import re
from typing import Any, Dict, List, Tuple

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
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse 3+ consecutive spaces into 1
    text = re.sub(r" {3,}", " ", text)
    # Remove repeated markdown horizontal rules (---, ___, ***)
    text = re.sub(r"([-_*]{3,}\n?){2,}", "---\n", text)
    # Deduplicate consecutive identical lines
    lines = text.split("\n")
    deduped = [lines[0]] if lines else []
    for line in lines[1:]:
        if line.strip() and line.strip() == deduped[-1].strip():
            continue  # skip duplicate line
        deduped.append(line)
    text = "\n".join(deduped)
    return text.strip()


def _clean_messages(messages: List[Dict]) -> List[Dict]:
    """Apply heuristic cleaning to all messages."""
    cleaned = []
    for msg in messages:
        cleaned.append(
            {
                **msg,
                "content": _clean_text(msg.get("content", "")),
            }
        )
    return cleaned


# ── Pass 2: Token-Aware Truncation & TF-IDF Compression ────────────────────────


def _compress_text_tfidf(text: str, max_tokens: int, model: str) -> str:
    """
    Smarter context compression using TF-IDF sentence importance ranking.
    Instead of dumb center-truncation, ranks sentences by informational density
    and retains the highest-importance sentences in their original chronological order.
    """
    current_tokens = count_tokens_in_string(text, model)
    if current_tokens <= max_tokens:
        return text

    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n{2,}', text) if s.strip()]
    if len(sentences) <= 2:
        return _truncate_text(text, max_tokens, model, use_tfidf=False)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
    except Exception:
        words = text.lower().split()
        word_freq = {}
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
        sentence_scores = [
            sum(word_freq.get(w, 0) for w in s.lower().split()) / max(len(s.split()), 1)
            for s in sentences
        ]

    total_sents = len(sentences)
    scored_indices = []
    for idx, (sent, score) in enumerate(zip(sentences, sentence_scores)):
        pos_multiplier = 1.0
        if idx == 0:
            pos_multiplier = 1.5
        elif idx == total_sents - 1:
            pos_multiplier = 1.3
        scored_indices.append((idx, float(score) * pos_multiplier, sent))

    scored_indices.sort(key=lambda x: x[1], reverse=True)

    selected_indices = []
    accumulated_tokens = 0
    separator_tokens = count_tokens_in_string(" ... ", model)

    for idx, score, sent in scored_indices:
        sent_tokens = count_tokens_in_string(sent, model)
        if accumulated_tokens + sent_tokens + separator_tokens <= max_tokens:
            selected_indices.append(idx)
            accumulated_tokens += sent_tokens + separator_tokens

    if not selected_indices:
        return _truncate_text(text, max_tokens, model, use_tfidf=False)

    selected_indices.sort()
    compressed_sentences = [sentences[i] for i in selected_indices]
    result = " ".join(compressed_sentences)

    logger.debug(
        "TF-IDF compression: %d → %d tokens (%d/%d sentences kept)",
        current_tokens,
        count_tokens_in_string(result, model),
        len(selected_indices),
        total_sents,
    )
    return result


def _truncate_text(text: str, max_tokens: int, model: str, use_tfidf: bool = True) -> str:
    """
    Truncate text to fit within max_tokens.
    When use_tfidf=True, ranks sentences by TF-IDF informational density.
    Falls back to edge-preserving center-truncation.
    """
    if use_tfidf:
        return _compress_text_tfidf(text, max_tokens, model)

    current_tokens = count_tokens_in_string(text, model)
    if current_tokens <= max_tokens:
        return text

    words = text.split()
    total_words = len(words)

    ratio = max_tokens / current_tokens
    keep_words = int(total_words * ratio)

    keep_start = int(keep_words * START_FRACTION)
    keep_end = int(keep_words * END_FRACTION)

    start_part = " ".join(words[:keep_start])
    end_part = " ".join(words[total_words - keep_end :])

    truncated = f"{start_part}\n\n...[CONTEXT COMPRESSED — {total_words - keep_start - keep_end} words removed]...\n\n{end_part}"
    logger.debug("Truncated message: %d → ~%d tokens", current_tokens, max_tokens)
    return truncated


def _truncate_messages(
    messages: List[Dict], token_budget: int, model: str, use_tfidf: bool = True
) -> List[Dict]:
    """
    Distribute the token budget across messages.
    - System messages: protect but tail-trim if necessary.
    - User/assistant messages: TF-IDF / center-truncate the largest ones first.
    """
    result = []

    system_msgs = [m for m in messages if m.get("role") == "system"]
    convo_msgs = [m for m in messages if m.get("role") != "system"]

    system_budget = int(token_budget * 0.30)
    convo_budget = token_budget - system_budget

    for msg in system_msgs:
        content = msg.get("content", "")
        token_count = count_tokens_in_string(content, model)
        if token_count > system_budget:
            truncated = _truncate_text(content, system_budget, model, use_tfidf=use_tfidf)
            result.append({**msg, "content": truncated})
        else:
            result.append(msg)

    if convo_msgs:
        last_msg = convo_msgs[-1]
        earlier_msgs = convo_msgs[:-1]
        last_tokens = count_tokens_in_string(last_msg.get("content", ""), model)
        earlier_budget = max(0, convo_budget - last_tokens)

        for msg in earlier_msgs:
            content = msg.get("content", "")
            token_count = count_tokens_in_string(content, model)
            per_msg_budget = max(15, earlier_budget // max(len(earlier_msgs), 1))
            if token_count > per_msg_budget:
                truncated = _truncate_text(content, per_msg_budget, model, use_tfidf=use_tfidf)
                result.append({**msg, "content": truncated})
            else:
                result.append(msg)

        result.append(last_msg)

    return result


# ── Main Entry Point ──────────────────────────────────────────────────────────


def compress(
    messages: List[Dict],
    model: str = "gpt-4o",
    max_tokens: int = MAX_TOKENS_THRESHOLD,
    mode: str = "smart",
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Run the full compression pipeline on a messages list.

    Args:
        messages: OpenAI-format messages list
        model: Model name (used for token counting)
        max_tokens: Token threshold above which compression is applied
        mode: "smart" (code/schema safe), "aggressive" (max savings), or "minimal" (cleaning only)

    Returns:
        (compressed_messages, stats)
    """
    original_tokens = count_tokens_in_messages(messages, model)

    cleaned = _clean_messages(messages)
    from app.engine.prompt_optimizer import deduplicate_messages, summarize_conversation

    cleaned, _ = deduplicate_messages(cleaned)

    if mode == "minimal":
        compressed_tokens = count_tokens_in_messages(cleaned, model)
        tokens_saved = max(0, original_tokens - compressed_tokens)
        return cleaned, {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": tokens_saved,
            "was_compressed": tokens_saved > 0,
            "compression_ratio": (
                round(1.0 - (compressed_tokens / original_tokens), 4)
                if original_tokens > 0
                else 0.0
            ),
        }

    effective_max_tokens = max_tokens
    if mode == "aggressive":
        effective_max_tokens = min(max_tokens, 1000)

    cleaned, _ = summarize_conversation(cleaned, max_turns=8)

    was_compressed = False
    final_messages = cleaned

    if original_tokens > effective_max_tokens:
        logger.info(
            "Compression triggered | original=%d tokens | threshold=%d | mode=%s",
            original_tokens,
            effective_max_tokens,
            mode,
        )
        was_compressed = True
        use_tfidf = mode in ("smart", "aggressive")
        final_messages = _truncate_messages(
            cleaned, effective_max_tokens, model, use_tfidf=use_tfidf
        )
    else:
        logger.debug(
            "No truncation needed | original=%d tokens (threshold=%d)",
            original_tokens,
            effective_max_tokens,
        )

    compressed_tokens = count_tokens_in_messages(final_messages, model)
    tokens_saved = max(0, original_tokens - compressed_tokens)
    compression_ratio = (
        round(1.0 - (compressed_tokens / original_tokens), 4)
        if original_tokens > 0
        else 0.0
    )

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
            original_tokens,
            compressed_tokens,
            tokens_saved,
            compression_ratio * 100,
        )

    return final_messages, stats
