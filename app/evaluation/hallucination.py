"""
HallucinationDetector — Factual Groundedness & Consistency Analyzer
Detects ungrounded assertions, numeric contradictions, and hallucination risk markers in responses.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("optillm.evaluation.hallucination")

_SPECULATIVE_MARKERS = [
    "i am guessing",
    "i cannot verify",
    "as an ai language model, i might be wrong",
    "i made this up",
    "unconfirmed rumor",
]


class HallucinationDetector:
    """
    Evaluates factual groundedness and consistency of LLM responses.
    """

    def detect(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
        reference_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect hallucination risk in response_text.

        Returns:
            {
                "hallucination_score": float,  # 0.0 = clean/grounded, 1.0 = heavy hallucination
                "risk_level": "low" | "medium" | "high",
                "explanation": str,
                "unsupported_numbers": List[str],
            }
        """
        if not response_text or not response_text.strip():
            return {
                "hallucination_score": 0.0,
                "risk_level": "low",
                "explanation": "Empty response — no hallucination detected.",
                "unsupported_numbers": [],
            }

        resp_lower = response_text.lower()
        score = 0.0
        explanations = []
        unsupported_nums = []

        # Check for speculative / ungrounded markers
        for marker in _SPECULATIVE_MARKERS:
            if marker in resp_lower:
                score += 0.35
                explanations.append(f"Detected speculative disclaimer: '{marker}'")

        # Check numeric grounding if reference context is provided
        if reference_context and reference_context.strip():
            ref_lower = reference_context.lower()
            resp_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", response_text))
            ref_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", reference_context))

            missing_nums = resp_numbers - ref_numbers
            if missing_nums:
                unsupported_nums = sorted(list(missing_nums))
                score += min(0.50, 0.15 * len(missing_nums))
                explanations.append(
                    f"Found {len(missing_nums)} numeric tokens in response not supported by reference context: {missing_nums}"
                )

        final_score = round(max(0.0, min(1.0, score)), 4)

        if final_score < 0.25:
            risk_level = "low"
        elif final_score < 0.60:
            risk_level = "medium"
        else:
            risk_level = "high"

        explanation_str = " | ".join(explanations) if explanations else "Response appears factually grounded."

        return {
            "hallucination_score": final_score,
            "risk_level": risk_level,
            "explanation": explanation_str,
            "unsupported_numbers": unsupported_nums,
        }


_global_detector: Optional[HallucinationDetector] = None


def get_hallucination_detector() -> HallucinationDetector:
    global _global_detector
    if _global_detector is None:
        _global_detector = HallucinationDetector()
    return _global_detector
