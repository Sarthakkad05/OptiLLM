"""
Model Router — Rule-Based Complexity Analyzer
Analyzes incoming prompts and selects the most cost-effective model
capable of handling the task.

Strategy:
  - Never UPGRADE a model (don't use GPT-4o when user asked for Flash)
  - Only DOWNGRADE from expensive to cheaper when task is simple enough
  - Score complexity across 3 dimensions: token count, keywords, conversation depth
  - Final score maps to LOW / MEDIUM / HIGH complexity → model tier

Routing Table:
  LOW    → gemini-2.0-flash    (cheapest, great for factual/simple tasks)
  MEDIUM → gpt-4o-mini         (balanced, good for analysis/explanation)
  HIGH   → keep requested model (don't interfere with complex tasks)

Savings:
  Savings = cost(original_model) - cost(routed_model)
  Savings are only positive when we route DOWN.
"""

import logging
from typing import List, Dict, Tuple
from enum import Enum
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.engine.router")


# ── Complexity Levels ─────────────────────────────────────────────────────────

class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Routing Table ─────────────────────────────────────────────────────────────
# Maps complexity → (preferred_model, provider)
ROUTING_TABLE: Dict[Complexity, Tuple[str, str]] = {
    Complexity.LOW:    ("gemini-2.0-flash", "gemini"),
    Complexity.MEDIUM: ("gpt-4o-mini",      "openai"),
    Complexity.HIGH:   (None, None),  # None = keep requested model
}

# ── Keyword Signals ───────────────────────────────────────────────────────────

# Strong indicators of SIMPLE tasks
_SIMPLE_KEYWORDS = {
    "what is", "what are", "who is", "when did", "where is",
    "define", "translate", "list", "summarize", "format",
    "convert", "extract", "calculate", "count", "spell",
    "yes or no", "true or false", "correct this",
}

# Strong indicators of COMPLEX tasks — these override simple signals
_COMPLEX_KEYWORDS = {
    "implement", "build", "architect", "design", "develop",
    "debug", "refactor", "optimize", "analyze", "compare",
    "evaluate", "explain in detail", "step by step",
    "write a function", "write code", "create a system",
    "multi-step", "reasoning", "prove", "derive",
}

# Models considered "expensive" that we can potentially downgrade
_EXPENSIVE_MODELS = {
    "gpt-4o", "gpt-4", "gpt-4-turbo",
    "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro",
}

# Models already cheap — don't touch them
_CHEAP_MODELS = {
    "gpt-4o-mini", "gpt-3.5-turbo",
    "gemini-2.0-flash", "gemini-1.5-flash",
}


# ── Scoring Functions ─────────────────────────────────────────────────────────

def _score_by_tokens(token_count: int) -> int:
    """
    Score based on prompt token count.
    Longer prompts usually require more capable models.
    Returns: 0 (simple), 1 (medium), 2 (complex)
    """
    if token_count < 80:
        return 0
    elif token_count < 400:
        return 1
    else:
        return 2


def _score_by_keywords(text: str) -> int:
    """
    Score based on keyword presence.
    Complex keywords dominate — they cancel out all simple signals.
    Returns: -1 (simple signal), 0 (neutral), 2 (complex signal)
    """
    text_lower = text.lower()

    # Complex keywords take priority
    for kw in _COMPLEX_KEYWORDS:
        if kw in text_lower:
            return 2

    # Simple keywords
    for kw in _SIMPLE_KEYWORDS:
        if kw in text_lower:
            return -1

    return 0


def _score_by_conversation_depth(messages: List[Dict]) -> int:
    """
    Score based on conversation history depth.
    Multi-turn conversations typically need more context tracking.
    Returns: 0 (single turn), 1 (short convo), 2 (long convo)
    """
    turns = len([m for m in messages if m.get("role") in ("user", "assistant")])
    if turns <= 1:
        return 0
    elif turns <= 4:
        return 1
    else:
        return 2


def _score_by_code_content(text: str) -> int:
    """Detect code blocks or technical content."""
    code_signals = ["```", "def ", "class ", "import ", "SELECT ", "function(", "=>"]
    for signal in code_signals:
        if signal in text:
            return 2
    return 0


# ── Main Routing Logic ────────────────────────────────────────────────────────

def analyze_complexity(messages: List[Dict], model: str) -> Tuple[Complexity, int, Dict]:
    """
    Analyze prompt complexity and return routing recommendation.

    Returns:
        (complexity_level, total_score, score_breakdown)
    """
    # Extract all text content for keyword analysis
    all_text = " ".join(m.get("content", "") for m in messages)
    user_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")

    token_count = count_tokens_in_messages(messages, model)

    # Score each dimension
    token_score = _score_by_tokens(token_count)
    keyword_score = _score_by_keywords(user_text)
    depth_score = _score_by_conversation_depth(messages)
    code_score = _score_by_code_content(all_text)

    total = token_score + keyword_score + depth_score + code_score

    breakdown = {
        "token_count": token_count,
        "token_score": token_score,
        "keyword_score": keyword_score,
        "depth_score": depth_score,
        "code_score": code_score,
        "total_score": total,
    }

    # Map score to complexity
    if total <= 0:
        complexity = Complexity.LOW
    elif total <= 2:
        complexity = Complexity.MEDIUM
    else:
        complexity = Complexity.HIGH

    return complexity, total, breakdown


def route(messages: List[Dict], requested_model: str) -> Dict:
    """
    Determine the optimal model for this request.

    Rules:
    1. If model is already cheap → keep it (no routing needed)
    2. If model is expensive AND task is LOW/MEDIUM → downgrade
    3. If task is HIGH complexity → keep requested model
    4. Never route to a MORE expensive model than requested

    Returns:
        {
            "model_used": str,
            "routed": bool,
            "complexity": str,
            "score": int,
            "score_breakdown": dict,
            "routing_reason": str,
        }
    """
    # Don't interfere if already on a cheap model
    if requested_model in _CHEAP_MODELS:
        return {
            "model_used": requested_model,
            "routed": False,
            "complexity": Complexity.MEDIUM,
            "score": 0,
            "score_breakdown": {},
            "routing_reason": "Model already cost-optimized — no routing applied.",
        }

    complexity, score, breakdown = analyze_complexity(messages, requested_model)
    routed_model, _ = ROUTING_TABLE.get(complexity, (None, None))

    logger.info(
        "Routing analysis | requested=%s | complexity=%s | score=%d | breakdown=%s",
        requested_model, complexity, score, breakdown,
    )

    # Only downgrade if task is simple/medium AND requested model is expensive
    if routed_model and requested_model in _EXPENSIVE_MODELS:
        logger.info(
            "Routing decision: %s → %s (complexity=%s, score=%d)",
            requested_model, routed_model, complexity, score,
        )
        return {
            "model_used": routed_model,
            "routed": True,
            "complexity": complexity,
            "score": score,
            "score_breakdown": breakdown,
            "routing_reason": f"Task complexity={complexity} — downgraded from {requested_model} to {routed_model}.",
        }

    # Keep original model for HIGH complexity
    logger.info(
        "No routing applied | model=%s | complexity=%s | score=%d",
        requested_model, complexity, score,
    )
    return {
        "model_used": requested_model,
        "routed": False,
        "complexity": complexity,
        "score": score,
        "score_breakdown": breakdown,
        "routing_reason": f"Task complexity={complexity} — original model retained.",
    }
