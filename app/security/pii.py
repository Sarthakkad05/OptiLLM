"""
PII Detection and Redaction Engine.
Integrates Microsoft Presidio (presidio-analyzer and presidio-anonymizer) with
NER (spaCy en_core_web_sm) and custom pattern recognizers.
Gracefully falls back to pure regex heuristics if Presidio or spaCy models are unavailable.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("optillm.security.pii")


class PIIDetector:
    """
    Dual-engine PII detection and redaction:
    1. Primary: Microsoft Presidio (NER + Rule-based analyzers with standardized operator replacement)
    2. Fallback: High-precision Regex patterns for SSN, Credit Cards, Emails, Phones, API Keys
    """

    REGEX_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        (
            "CREDIT_CARD",
            re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
            "[REDACTED_CREDIT_CARD]",
        ),
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "[REDACTED_EMAIL]",
        ),
        (
            "PHONE",
            re.compile(r"\b(?:\+?1[-. ]?)?\(?[2-9]\d{2}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
            "[REDACTED_PHONE]",
        ),
        (
            "API_KEY",
            re.compile(r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b"),
            "[REDACTED_API_KEY]",
        ),
    ]

    def __init__(self, enable_presidio: Optional[bool] = None, model_name: Optional[str] = None):
        self.enable_presidio = (
            enable_presidio if enable_presidio is not None else settings.PRESIDIO_ENABLED
        )
        self.model_name = model_name or settings.PRESIDIO_MODEL

        self.analyzer = None
        self.anonymizer = None
        self.operators = {}
        self._presidio_initialized = False

        if self.enable_presidio:
            self._init_presidio()

    def _init_presidio(self):
        """Attempts to initialize Presidio analyzer and anonymizer engines."""
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig

            # Configure spaCy model engine
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": self.model_name}],
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = provider.create_engine()

            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()

            # Standardized enterprise redaction operators
            self.operators = {
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_EMAIL]"}),
                "EMAIL": OperatorConfig("replace", {"new_value": "[REDACTED_EMAIL]"}),
                "US_SSN": OperatorConfig("replace", {"new_value": "[REDACTED_SSN]"}),
                "SSN": OperatorConfig("replace", {"new_value": "[REDACTED_SSN]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED_PHONE]"}),
                "PHONE": OperatorConfig("replace", {"new_value": "[REDACTED_PHONE]"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[REDACTED_CREDIT_CARD]"}),
                "API_KEY": OperatorConfig("replace", {"new_value": "[REDACTED_API_KEY]"}),
                "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_PERSON]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED_LOCATION]"}),
                "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_IP]"}),
                "IBAN_CODE": OperatorConfig("replace", {"new_value": "[REDACTED_IBAN]"}),
                "CRYPTO": OperatorConfig("replace", {"new_value": "[REDACTED_CRYPTO]"}),
            }

            # Register custom API Key recognizer
            api_pattern = Pattern(
                name="api_key_pattern",
                regex=r"\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b",
                score=0.95,
            )
            api_recognizer = PatternRecognizer(
                supported_entity="API_KEY", patterns=[api_pattern]
            )
            self.analyzer.registry.add_recognizer(api_recognizer)

            # Register loose SSN pattern
            ssn_pattern = Pattern(
                name="loose_ssn_pattern",
                regex=r"\b\d{3}-\d{2}-\d{4}\b",
                score=0.95,
            )
            ssn_recognizer = PatternRecognizer(
                supported_entity="SSN", patterns=[ssn_pattern]
            )
            self.analyzer.registry.add_recognizer(ssn_recognizer)

            self._presidio_initialized = True
            logger.info("Presidio PII Detector initialized successfully with model '%s'", self.model_name)
        except Exception as exc:
            logger.warning(
                "Failed to initialize Presidio (%s) — falling back to regex engine.", exc
            )
            self.analyzer = None
            self.anonymizer = None
            self._presidio_initialized = False

    @property
    def is_presidio_active(self) -> bool:
        return self._presidio_initialized and self.analyzer is not None

    def get_engine_type(self) -> str:
        return "presidio" if self.is_presidio_active else "regex"

    def analyze(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """
        Analyzes text and returns detailed list of identified PII entities.
        """
        if not text:
            return []

        entities = []
        if self.is_presidio_active:
            try:
                presidio_results = self.analyzer.analyze(text=text, language=language)
                for r in presidio_results:
                    matched_text = text[r.start : r.end]
                    # Filter false-positive acronyms tagged as ORGANIZATION
                    if r.entity_type == "ORGANIZATION" and matched_text.strip().upper() in {"SSN", "API", "KEY", "ID"}:
                        continue
                    entities.append({
                        "entity_type": r.entity_type,
                        "start": r.start,
                        "end": r.end,
                        "score": r.score,
                        "text": matched_text,
                        "source": "presidio",
                    })
                return entities
            except Exception as exc:
                logger.warning("Presidio analysis error (%s) — falling back to regex", exc)

        # Regex fallback
        for pii_type, pattern, _ in self.REGEX_PATTERNS:
            for match in pattern.finditer(text):
                entities.append({
                    "entity_type": pii_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.95,
                    "text": match.group(),
                    "source": "regex",
                })

        return entities

    def redact(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Scans text for PII and returns redacted output.
        Returns:
            {
                "redacted_text": str,
                "pii_found": bool,
                "detected_types": List[str],
            }
        """
        if not text:
            return {"redacted_text": "", "pii_found": False, "detected_types": []}

        detected_types = set()

        if self.is_presidio_active:
            try:
                results = self.analyzer.analyze(text=text, language=language)
                # Filter out false-positive acronyms tagged as ORGANIZATION
                filtered_results = [
                    r for r in results
                    if not (r.entity_type == "ORGANIZATION" and text[r.start : r.end].strip().upper() in {"SSN", "API", "KEY", "ID"})
                ]

                if filtered_results:
                    for r in filtered_results:
                        detected_types.add(r.entity_type)
                        if r.entity_type == "EMAIL_ADDRESS":
                            detected_types.add("EMAIL")
                        elif r.entity_type == "US_SSN":
                            detected_types.add("SSN")
                        elif r.entity_type == "PHONE_NUMBER":
                            detected_types.add("PHONE")

                    anonymized_result = self.anonymizer.anonymize(
                        text=text, analyzer_results=filtered_results, operators=self.operators
                    )
                    redacted_text = anonymized_result.text
                else:
                    redacted_text = text

                # Supplementary regex pass for any edge cases
                for pii_type, pattern, placeholder in self.REGEX_PATTERNS:
                    if pattern.search(redacted_text):
                        detected_types.add(pii_type)
                        redacted_text = pattern.sub(placeholder, redacted_text)

                return {
                    "redacted_text": redacted_text,
                    "pii_found": len(detected_types) > 0,
                    "detected_types": sorted(list(detected_types)),
                }
            except Exception as exc:
                logger.warning("Presidio redaction failed (%s) — using regex fallback", exc)
                return self._regex_redact(text)

        return self._regex_redact(text)

    def _regex_redact(self, text: str) -> Dict[str, Any]:
        """Fallback regex redaction method."""
        redacted_text = text
        detected_types = []

        for pii_type, pattern, placeholder in self.REGEX_PATTERNS:
            matches = pattern.findall(redacted_text)
            if matches:
                detected_types.append(pii_type)
                redacted_text = pattern.sub(placeholder, redacted_text)

        return {
            "redacted_text": redacted_text,
            "pii_found": len(detected_types) > 0,
            "detected_types": detected_types,
        }


_global_pii_detector: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    """Returns singleton instance of PIIDetector."""
    global _global_pii_detector
    if _global_pii_detector is None:
        _global_pii_detector = PIIDetector()
    return _global_pii_detector
