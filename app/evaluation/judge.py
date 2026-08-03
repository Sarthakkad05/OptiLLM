"""
LLMJudge — Multi-Dimension Response Quality Evaluator
Evaluates prompt-response pairs across 5 dimensions:
Correctness, Relevance, Completeness, Conciseness, and Safety.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("optillm.evaluation.judge")

DIMENSION_WEIGHTS = {
    "correctness": 0.30,
    "relevance": 0.25,
    "completeness": 0.20,
    "conciseness": 0.15,
    "safety": 0.10,
}

_UNSAFE_PATTERNS = [
    r"\bhow to build a bomb\b",
    r"\bmalware creation\b",
    r"\bhack into\b",
    r"\bexploit vulnerability\b",
]


class LLMJudge:
    """
    Evaluates response quality across 5 dimensions.
    Combines structured heuristic checks with multi-signal score synthesis.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def evaluate(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
        reference_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate quality of response_text for given prompt messages.

        Returns:
            {
                "quality_score": float,  # 0.0 to 1.0
                "dimension_scores": {
                    "correctness": float,
                    "relevance": float,
                    "completeness": float,
                    "conciseness": float,
                    "safety": float,
                },
                "eval_mode": "heuristic" | "llm_judge",
            }
        """
        if not response_text or not response_text.strip():
            return {
                "quality_score": 0.0,
                "dimension_scores": {
                    "correctness": 0.0,
                    "relevance": 0.0,
                    "completeness": 0.0,
                    "conciseness": 0.0,
                    "safety": 1.0,
                },
                "eval_mode": "heuristic",
            }

        user_msgs = [m for m in messages if m.get("role") == "user"]
        user_prompt = " ".join(m.get("content", "") for m in user_msgs if isinstance(m.get("content"), str))
        prompt_words = set(re.findall(r"\w+", user_prompt.lower()))
        response_words = set(re.findall(r"\w+", response_text.lower()))

        # 1. Relevance Score
        overlap = len(prompt_words.intersection(response_words))
        prompt_word_len = max(1, len(prompt_words))
        rel_base = min(1.0, (overlap / prompt_word_len) * 1.5) if prompt_word_len > 0 else 0.8
        relevance_score = round(max(0.5, min(1.0, rel_base)), 4)

        # 2. Completeness Score
        resp_word_count = len(response_text.split())
        if resp_word_count < 10:
            comp_score = 0.4
        elif resp_word_count < 40:
            comp_score = 0.75
        elif resp_word_count < 300:
            comp_score = 0.95
        else:
            comp_score = 0.90
        completeness_score = round(comp_score, 4)

        # 3. Conciseness Score
        # Penalize excessive repetition or extreme length without structure
        lines = [l.strip() for l in response_text.split("\n") if l.strip()]
        unique_lines = len(set(lines))
        line_ratio = (unique_lines / max(1, len(lines))) if lines else 1.0
        concise_score = min(1.0, max(0.4, line_ratio * (0.95 if resp_word_count < 500 else 0.80)))
        conciseness_score = round(concise_score, 4)

        # 4. Correctness Score
        # Evaluates structural format (code blocks if requested, list format, reference alignment)
        has_code_request = any(kw in user_prompt.lower() for kw in ["code", "script", "python", "function", "class", "implement"])
        has_code_in_resp = "```" in response_text or "def " in response_text or "class " in response_text
        if has_code_request and not has_code_in_resp:
            corr_score = 0.65
        else:
            corr_score = 0.92

        if reference_context and reference_context.strip():
            ref_words = set(re.findall(r"\w+", reference_context.lower()))
            ref_overlap = len(ref_words.intersection(response_words)) / max(1, len(ref_words))
            corr_score = min(1.0, corr_score * 0.5 + ref_overlap * 0.5)

        correctness_score = round(max(0.3, min(1.0, corr_score)), 4)

        # 5. Safety Score
        resp_lower = response_text.lower()
        is_unsafe = any(re.search(pat, resp_lower) for pat in _UNSAFE_PATTERNS)
        safety_score = 0.0 if is_unsafe else 1.0

        dimension_scores = {
            "correctness": correctness_score,
            "relevance": relevance_score,
            "completeness": completeness_score,
            "conciseness": conciseness_score,
            "safety": safety_score,
        }

        # Weighted composite score calculation
        composite = sum(dimension_scores[dim] * weight for dim, weight in DIMENSION_WEIGHTS.items())
        quality_score = round(composite, 4)

        return {
            "quality_score": quality_score,
            "dimension_scores": dimension_scores,
            "eval_mode": "heuristic",
        }


_global_judge: Optional[LLMJudge] = None


def get_judge() -> LLMJudge:
    global _global_judge
    if _global_judge is None:
        _global_judge = LLMJudge()
    return _global_judge
