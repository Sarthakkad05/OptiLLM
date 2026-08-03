"""
PII Detection and Redaction Engine
Detects and redacts sensitive personal data (SSN, credit cards, emails, phone numbers, API keys)
from request payloads before provider dispatch.
"""

import re
from typing import Any, Dict, List, Tuple


class PIIDetector:
    """
    Regex & heuristic engine to scan text for PII data and return redacted output.
    """

    PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        ("CREDIT_CARD", re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"), "[REDACTED_CREDIT_CARD]"),
        ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        ("PHONE", re.compile(r"\b(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b"), "[REDACTED_PHONE]"),
        ("API_KEY", re.compile(r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b"), "[REDACTED_API_KEY]"),
    ]

    def redact(self, text: str) -> Dict[str, Any]:
        """
        Scans text for PII patterns and replaces them with redacted placeholders.

        Returns:
            {
                "redacted_text": str,
                "pii_found": bool,
                "detected_types": List[str],
            }
        """
        if not text:
            return {"redacted_text": "", "pii_found": False, "detected_types": []}

        redacted_text = text
        detected_types = []

        for pii_type, pattern, placeholder in self.PATTERNS:
            matches = pattern.findall(redacted_text)
            if matches:
                detected_types.append(pii_type)
                redacted_text = pattern.sub(placeholder, redacted_text)

        return {
            "redacted_text": redacted_text,
            "pii_found": len(detected_types) > 0,
            "detected_types": detected_types,
        }


_global_pii_detector: PIIDetector | None = None


def get_pii_detector() -> PIIDetector:
    global _global_pii_detector
    if _global_pii_detector is None:
        _global_pii_detector = PIIDetector()
    return _global_pii_detector
