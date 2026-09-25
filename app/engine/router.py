"""
Model Router — Complexity Analyzer & Intelligent AI Router
Analyzes incoming prompts and selects the most cost-effective model
capable of handling the task.

Routing Modes (settings.ROUTING_MODE):
  - rule_based: Rule-based score across tokens, keywords, and conversation depth.
  - ai: ML classifier (`AIRouter`) with fallback to rule-based when confidence < threshold.
  - shadow: Runs both rule-based and AI router in parallel, uses rule-based for execution,
            and logs shadow disagreement metrics.

Routing Table:
  LOW    → gemini-2.0-flash    (cheapest, great for factual/simple tasks)
  MEDIUM → gpt-4o-mini         (balanced, good for analysis/explanation)
  HIGH   → keep requested model (don't interfere with complex tasks)
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.engine.ai_router import get_ai_router
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.engine.router")


# ── Complexity Levels ─────────────────────────────────────────────────────────


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ── Routing Table ─────────────────────────────────────────────────────────────
ROUTING_TABLE: Dict[Complexity, Tuple[str, str]] = {
    Complexity.LOW: ("gemini-2.0-flash", "gemini"),
    Complexity.MEDIUM: ("gpt-4o-mini", "openai"),
    Complexity.HIGH: (None, None),  # None = keep requested model
}

# ── Keyword Signals ───────────────────────────────────────────────────────────

_SIMPLE_KEYWORDS = {
    "what is", "what are", "who is", "when did", "where is", "define",
    "translate", "list", "summarize", "format", "convert", "extract",
    "calculate", "count", "spell", "yes or no", "true or false", "correct this"
}

_COMPLEX_KEYWORDS = {
    "implement", "build", "architect", "design", "develop", "debug",
    "refactor", "optimize", "analyze", "compare", "evaluate", "explain in detail",
    "step by step", "write a function", "write code", "create a system",
    "multi-step", "reasoning", "prove", "derive"
}

_EXPENSIVE_MODELS = {
    "gpt-4o", "gpt-4", "gpt-4-turbo", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"
}

_CHEAP_MODELS = {
    "gpt-4o-mini", "gpt-3.5-turbo", "gemini-2.0-flash", "gemini-1.5-flash"
}


# ── Scoring Functions ─────────────────────────────────────────────────────────


def _score_by_tokens(token_count: int) -> int:
    if token_count < 80:
        return 0
    elif token_count < 400:
        return 1
    else:
        return 2


def _score_by_keywords(text: str) -> int:
    text_lower = text.lower()
    for kw in _COMPLEX_KEYWORDS:
        if kw in text_lower:
            return 2
    for kw in _SIMPLE_KEYWORDS:
        if kw in text_lower:
            return -1
    return 0


def _score_by_conversation_depth(messages: List[Dict]) -> int:
    turns = len([m for m in messages if m.get("role") in ("user", "assistant")])
    if turns <= 1:
        return 0
    elif turns <= 4:
        return 1
    else:
        return 2


def _score_by_code_content(text: str) -> int:
    code_signals = ["```", "def ", "class ", "import ", "SELECT ", "function(", "=>"]
    for signal in code_signals:
        if signal in text:
            return 2
    return 0


# ── Main Routing Logic ────────────────────────────────────────────────────────


def analyze_complexity(
    messages: List[Dict], model: str
) -> Tuple[Complexity, int, Dict]:
    """
    Analyze prompt complexity using rule-based heuristic scoring.
    """
    all_text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)
    )

    token_count = count_tokens_in_messages(messages, model)

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

    if total <= 0:
        complexity = Complexity.LOW
    elif total <= 2:
        complexity = Complexity.MEDIUM
    else:
        complexity = Complexity.HIGH

    return complexity, total, breakdown


def route(messages: List[Dict], requested_model: str) -> Dict:
    """
    Determine the optimal model for this request based on ROUTING_MODE.
    """
    from app.core.config import settings

    mode = (settings.ROUTING_MODE or "shadow").lower()

    if requested_model in _CHEAP_MODELS:
        return {
            "model_used": requested_model,
            "routed": False,
            "complexity": Complexity.MEDIUM,
            "score": 0,
            "score_breakdown": {},
            "routing_reason": "Model already cost-optimized — no routing applied.",
            "routing_mode": mode,
            "confidence": 1.0,
            "shadow_disagreement": False,
            "ai_predicted_complexity": None,
        }

    rule_complexity, score, breakdown = analyze_complexity(messages, requested_model)
    ai_router = get_ai_router()
    ai_pred = ai_router.predict(messages, requested_model)

    final_complexity = rule_complexity
    used_ai = False
    confidence_fallback = False
    shadow_disagreement = False

    if mode == "ai":
        conf_threshold = settings.AI_ROUTER_CONFIDENCE_THRESHOLD
        if ai_pred["confidence"] >= conf_threshold:
            final_complexity = Complexity(ai_pred["predicted_complexity"])
            used_ai = True
        else:
            confidence_fallback = True
            logger.info(
                "AI router confidence (%.2f) below threshold (%.2f). Falling back to rule-based complexity (%s).",
                ai_pred["confidence"],
                conf_threshold,
                rule_complexity,
            )
    elif mode == "shadow":
        ai_complexity_str = ai_pred["predicted_complexity"]
        rule_complexity_str = rule_complexity.value
        shadow_disagreement = (ai_complexity_str != rule_complexity_str)
        if shadow_disagreement:
            logger.warning(
                "Shadow mode disagreement: Rule=%s vs AI=%s (conf=%.2f)",
                rule_complexity_str,
                ai_complexity_str,
                ai_pred["confidence"],
            )

    routed_model, provider = ROUTING_TABLE.get(final_complexity, (None, None))

    if provider == "gemini":
        gemini_missing = (
            not settings.GEMINI_API_KEY
            or settings.GEMINI_API_KEY == "your_gemini_api_key_here"
        )
        if gemini_missing:
            routed_model = "gpt-4o-mini"

    if routed_model and requested_model in _EXPENSIVE_MODELS:
        reason = (
            f"Task complexity={final_complexity.value} ({mode} mode"
            f"{', fallback applied' if confidence_fallback else ''}) — "
            f"downgraded from {requested_model} to {routed_model}."
        )
        return {
            "model_used": routed_model,
            "routed": True,
            "complexity": final_complexity,
            "score": score,
            "score_breakdown": breakdown,
            "routing_reason": reason,
            "routing_mode": mode,
            "confidence": ai_pred["confidence"],
            "shadow_disagreement": shadow_disagreement,
            "confidence_fallback": confidence_fallback,
            "ai_probabilities": ai_pred["probabilities"],
            "ai_predicted_complexity": ai_pred["predicted_complexity"],
        }

    return {
        "model_used": requested_model,
        "routed": False,
        "complexity": final_complexity,
        "score": score,
        "score_breakdown": breakdown,
        "routing_reason": f"Task complexity={final_complexity.value} — original model retained.",
        "routing_mode": mode,
        "confidence": ai_pred["confidence"],
        "shadow_disagreement": shadow_disagreement,
        "confidence_fallback": confidence_fallback,
        "ai_predicted_complexity": ai_pred["predicted_complexity"],
        "ai_probabilities": ai_pred["probabilities"],
    }


def explain_routing(messages: List[Dict], requested_model: str) -> Dict[str, Any]:
    """
    Generate complete explainability report for a prompt and requested model.
    Exposed via POST /api/v1/router/explain.
    """
    from app.core.config import settings

    mode = (settings.ROUTING_MODE or "shadow").lower()
    rule_comp, score, breakdown = analyze_complexity(messages, requested_model)
    ai_router = get_ai_router()
    ai_pred = ai_router.predict(messages, requested_model)

    routing_res = route(messages, requested_model)

    return {
        "requested_model": requested_model,
        "routing_mode": mode,
        "confidence_threshold": settings.AI_ROUTER_CONFIDENCE_THRESHOLD,
        "selected_complexity": routing_res["complexity"],
        "decision": {
            "model_used": routing_res["model_used"],
            "routed": routing_res["routed"],
            "reason": routing_res["routing_reason"],
        },
        "rule_analysis": {
            "complexity": rule_comp.value,
            "score": score,
            "breakdown": breakdown,
        },
        "ai_analysis": {
            "predicted_complexity": ai_pred["predicted_complexity"],
            "confidence": ai_pred["confidence"],
            "probabilities": ai_pred["probabilities"],
            "is_trained_model": ai_pred["is_trained_model"],
            "feature_dict": ai_pred["feature_dict"],
            "feature_importances": ai_router.get_feature_importances(),
        },
        "shadow_mode": {
            "active": (mode == "shadow"),
            "disagreement": (rule_comp.value != ai_pred["predicted_complexity"]),
            "rule_complexity": rule_comp.value,
            "ai_complexity": ai_pred["predicted_complexity"],
        },
        "fallback": {
            "occurred": routing_res.get("confidence_fallback", False),
            "reason": (
                f"AI confidence {ai_pred['confidence']:.2f} < threshold {settings.AI_ROUTER_CONFIDENCE_THRESHOLD:.2f}"
                if routing_res.get("confidence_fallback", False)
                else None
            ),
        },
    }


# ── Runtime Config ────────────────────────────────────────────────────────────


def get_routing_config() -> Dict:
    """
    Returns current routing configuration, mode, and thresholds.
    """
    from app.core.config import settings

    return {
        "routing_mode": settings.ROUTING_MODE,
        "confidence_threshold": settings.AI_ROUTER_CONFIDENCE_THRESHOLD,
        "ai_model_path": settings.AI_ROUTER_MODEL_PATH,
        "is_ai_model_trained": get_ai_router().is_trained,
        "routing_table": {
            level.value: model for level, (model, _) in ROUTING_TABLE.items() if model
        },
        "score_thresholds": {
            "low_max_score": 0,
            "medium_max_score": 2,
        },
        "expensive_models": list(_EXPENSIVE_MODELS),
        "cheap_models": list(_CHEAP_MODELS),
    }


def update_routing_config(
    low_model: Optional[str] = None,
    medium_model: Optional[str] = None,
    routing_mode: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
) -> Dict:
    """
    Update routing table, routing mode, or confidence threshold at runtime.
    """
    from app.core.config import settings

    if low_model is not None:
        provider = "gemini" if "gemini" in low_model.lower() else "openai"
        ROUTING_TABLE[Complexity.LOW] = (low_model, provider)
        logger.info("Routing config updated: LOW → %s (%s)", low_model, provider)

    if medium_model is not None:
        provider = "gemini" if "gemini" in medium_model.lower() else "openai"
        ROUTING_TABLE[Complexity.MEDIUM] = (medium_model, provider)
        logger.info("Routing config updated: MEDIUM → %s (%s)", medium_model, provider)

    if routing_mode is not None:
        valid_modes = {"rule_based", "ai", "shadow"}
        if routing_mode.lower() not in valid_modes:
            raise ValueError(f"Invalid routing_mode: {routing_mode}. Must be one of {valid_modes}")
        settings.ROUTING_MODE = routing_mode.lower()
        logger.info("Routing mode updated: %s", settings.ROUTING_MODE)

    if confidence_threshold is not None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        settings.AI_ROUTER_CONFIDENCE_THRESHOLD = confidence_threshold
        logger.info("AI Router confidence threshold updated: %.2f", confidence_threshold)

    return get_routing_config()
