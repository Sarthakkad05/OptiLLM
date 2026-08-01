"""
Prompt Optimization Engine.
Provides conversation summarization and duplicate message removal
to reduce prompt token consumption by 30-60%.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("optillm.engine.prompt_optimizer")


def deduplicate_messages(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Strips consecutive identical messages (same role and exact same content).
    Returns (deduplicated_messages, removed_count).
    """
    if not messages or len(messages) <= 1:
        return messages, 0

    deduped = []
    removed_count = 0

    for msg in messages:
        if (
            deduped
            and deduped[-1].get("role") == msg.get("role")
            and deduped[-1].get("content", "").strip() == msg.get("content", "").strip()
        ):
            removed_count += 1
            continue
        deduped.append(msg)

    if removed_count > 0:
        logger.info(
            "Deduplicated prompt messages | removed=%d duplicates", removed_count
        )

    return deduped, removed_count


def summarize_conversation(
    messages: List[Dict], max_turns: int = 8
) -> Tuple[List[Dict], bool]:
    """
    Condenses long multi-turn conversations exceeding max_turns.

    Strategy:
      - Preserve initial system prompt if present.
      - Take older conversation turns (excluding final 3 turns) and condense into a summary block.
      - Retain recent 3 turns intact to preserve immediate chat context.
    """
    non_system = [m for m in messages if m.get("role") != "system"]
    system_msgs = [m for m in messages if m.get("role") == "system"]

    # Check if turn count exceeds threshold
    if len(non_system) <= max_turns:
        return messages, False

    # Split into older turns to condense and recent turns to keep (last 3)
    older_turns = non_system[:-3]
    recent_turns = non_system[-3:]

    # Build factual summary text from older turns
    summary_snippets = []
    for m in older_turns:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")[:150]
        summary_snippets.append(f"{role}: {content}")

    summary_content = (
        f"[CONVERSATION SUMMARY ({len(older_turns)} older turns condensed):\n"
        + "\n".join(summary_snippets)
        + "\n]"
    )

    summary_message = {"role": "system", "content": summary_content}

    optimized = []
    if system_msgs:
        optimized.extend(system_msgs)
    optimized.append(summary_message)
    optimized.extend(recent_turns)

    logger.info(
        "Summarized conversation | original_turns=%d → condensed_turns=%d",
        len(messages),
        len(optimized),
    )

    return optimized, True
