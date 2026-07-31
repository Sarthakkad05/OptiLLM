"""
usecase/support_bot/knowledge_base.py
======================================
Warm the OptiLLM semantic cache with FAQ question-answer pairs at startup.

When a user asks a question similar to one already cached, the gateway
returns it in ~10ms at zero cost — no LLM call needed.

This is the "seed cache" pattern: pre-load known question-answer pairs so
cache hits are available from the very first user message.
"""

import os
import re
import time
import logging
import requests
from typing import List, Tuple

logger = logging.getLogger("support_bot.knowledge_base")

GATEWAY_URL = os.environ.get("OPTILLM_URL", "http://localhost:8000")
MODEL        = "gpt-4o"


def _parse_faq(path: str) -> List[Tuple[str, str]]:
    """Parse Q&A pairs from a markdown FAQ file."""
    pairs = []
    with open(path, "r") as f:
        content = f.read()

    # Match bold Q: / A: patterns
    # Pattern: **Q: question** followed by answer paragraph(s) until next **Q:
    blocks = re.split(r'\n(?=\*\*Q:)', content)
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        first = lines[0].strip()
        match = re.match(r'\*\*Q:\s*(.+?)\*\*', first)
        if not match:
            continue
        question = match.group(1).strip()
        answer_lines = []
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("**Q:"):
                break
            if stripped:
                answer_lines.append(stripped)
        if answer_lines:
            pairs.append((question, " ".join(answer_lines)))

    return pairs


def _seed_question(question: str, answer: str, system_prompt: str) -> bool:
    """
    Send a Q&A pair to the gateway to seed the cache.
    The gateway will call the LLM (or use mock mode) and cache the result.
    After this, semantically similar questions will hit the cache.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        # Pre-load the known answer by injecting it as a user→assistant pair
        # Actually: we just make the real call and let OptiLLM cache the result.
        # No special injection needed — the cache works on user message similarity.
    }
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/v1/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Failed to seed question '%s': %s", question[:60], e)
        return False


def warm_cache(faq_path: str, system_prompt: str, max_seed: int = 10) -> int:
    """
    Seed the OptiLLM cache with the most important FAQ questions.

    Args:
        faq_path: Path to the FAQ markdown file.
        system_prompt: System prompt to use when seeding.
        max_seed: Maximum number of questions to seed (cost control).

    Returns:
        Number of questions successfully cached.
    """
    pairs = _parse_faq(faq_path)
    if not pairs:
        logger.warning("No Q&A pairs found in %s", faq_path)
        return 0

    seed_pairs = pairs[:max_seed]
    logger.info("Warming cache with %d/%d FAQ questions...", len(seed_pairs), len(pairs))

    seeded = 0
    for i, (question, _expected_answer) in enumerate(seed_pairs):
        logger.info("  [%d/%d] Seeding: %s", i + 1, len(seed_pairs), question[:70])
        if _seed_question(question, _expected_answer, system_prompt):
            seeded += 1
        time.sleep(0.1)  # Small delay to avoid hammering the gateway

    logger.info("Cache warm-up complete: %d/%d questions seeded.", seeded, len(seed_pairs))
    return seeded
