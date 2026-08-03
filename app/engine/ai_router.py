"""
AI Router — ML-Based Prompt Complexity Classifier
Extracts numerical feature vectors from prompt messages and uses an ML classifier
to predict prompt complexity (low, medium, high) with confidence scores.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.core.config import settings
from app.services.token_counter import count_tokens_in_messages

logger = logging.getLogger("optillm.engine.ai_router")

# Standardized feature names used by the classifier
FEATURE_NAMES = [
    "token_count",
    "char_count",
    "turn_count",
    "user_msg_count",
    "complex_keyword_count",
    "simple_keyword_count",
    "code_signal_count",
    "avg_tokens_per_turn",
    "is_expensive_model",
]

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

_CODE_SIGNALS = ["```", "def ", "class ", "import ", "SELECT ", "function(", "=>"]

_EXPENSIVE_MODELS = {
    "gpt-4o", "gpt-4", "gpt-4-turbo", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"
}

COMPLEXITY_CLASSES = ["low", "medium", "high"]


def extract_features(messages: List[Dict], model: str) -> Dict[str, float]:
    """
    Extract numerical feature dictionary from prompt messages and model.
    """
    all_text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)
    ).lower()

    token_count = float(count_tokens_in_messages(messages, model))
    char_count = float(len(all_text))
    turn_count = float(len([m for m in messages if m.get("role") in ("user", "assistant")]))
    user_msg_count = float(len([m for m in messages if m.get("role") == "user"]))

    complex_count = float(sum(1 for kw in _COMPLEX_KEYWORDS if kw in user_text))
    simple_count = float(sum(1 for kw in _SIMPLE_KEYWORDS if kw in user_text))
    code_count = float(sum(1 for signal in _CODE_SIGNALS if signal in all_text))

    avg_tokens = token_count / max(1.0, turn_count)
    is_expensive = 1.0 if model.lower() in _EXPENSIVE_MODELS else 0.0

    return {
        "token_count": token_count,
        "char_count": char_count,
        "turn_count": turn_count,
        "user_msg_count": user_msg_count,
        "complex_keyword_count": complex_count,
        "simple_keyword_count": simple_count,
        "code_signal_count": code_count,
        "avg_tokens_per_turn": avg_tokens,
        "is_expensive_model": is_expensive,
    }


def extract_feature_vector(messages: List[Dict], model: str) -> List[float]:
    """
    Extract feature vector matching FEATURE_NAMES ordering.
    """
    feat_dict = extract_features(messages, model)
    return [feat_dict[name] for name in FEATURE_NAMES]


class AIRouter:
    """
    ML Classifier for AI Routing.
    Wraps a RandomForestClassifier with fallback heuristic probability calibration.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.AI_ROUTER_MODEL_PATH
        self.clf: Optional[RandomForestClassifier] = None
        self.is_trained: bool = False
        self.load_model()

    def load_model(self) -> bool:
        """Load trained scikit-learn model from disk if present."""
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                if isinstance(data, dict) and "clf" in data:
                    self.clf = data["clf"]
                    self.is_trained = True
                elif hasattr(data, "predict"):
                    self.clf = data
                    self.is_trained = True
                logger.info("Loaded AI Router model from %s", self.model_path)
                return True
            except Exception as e:
                logger.warning("Failed to load AI Router model from %s: %s", self.model_path, e)
                self.is_trained = False
                self.clf = None
        return False

    def save_model(self) -> bool:
        """Save current trained model to disk."""
        if not self.clf or not self.is_trained:
            logger.warning("Cannot save model: AI Router is not trained.")
            return False
        try:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            joblib.dump({"clf": self.clf, "feature_names": FEATURE_NAMES}, self.model_path)
            logger.info("Saved AI Router model to %s", self.model_path)
            return True
        except Exception as e:
            logger.error("Failed to save AI Router model: %s", e)
            return False

    def predict(self, messages: List[Dict], model: str) -> Dict[str, Any]:
        """
        Predict prompt complexity level and confidence.

        Returns:
            {
                "predicted_complexity": "low" | "medium" | "high",
                "confidence": float,
                "probabilities": {"low": float, "medium": float, "high": float},
                "feature_dict": dict,
                "is_trained_model": bool,
            }
        """
        feat_dict = extract_features(messages, model)
        feat_vec = [feat_dict[name] for name in FEATURE_NAMES]

        if self.is_trained and self.clf is not None:
            try:
                probs = self.clf.predict_proba([feat_vec])[0]
                classes = list(self.clf.classes_)
                prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
                # Ensure all 3 complexity levels exist in prob_dict
                for c in COMPLEXITY_CLASSES:
                    if c not in prob_dict:
                        prob_dict[c] = 0.0

                top_class = max(prob_dict, key=prob_dict.get)
                confidence = prob_dict[top_class]

                return {
                    "predicted_complexity": top_class,
                    "confidence": round(confidence, 4),
                    "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
                    "feature_dict": feat_dict,
                    "is_trained_model": True,
                }
            except Exception as e:
                logger.error("Error during ML prediction, falling back to heuristic: %s", e)

        # Baseline heuristic calibrated probability prediction when no ML model is saved
        return self._heuristic_prediction(feat_dict)

    def _heuristic_prediction(self, feat: Dict[str, float]) -> Dict[str, Any]:
        """
        Deterministic, smoothly-calibrated baseline probability prediction.
        Calculates a continuous complexity score and maps to class probabilities via softmax.
        """
        score = (
            (0.8 if feat["token_count"] > 400 else (0.4 if feat["token_count"] > 80 else 0.0))
            + feat["complex_keyword_count"] * 1.5
            - feat["simple_keyword_count"] * 1.0
            + feat["code_signal_count"] * 1.5
            + (0.5 if feat["turn_count"] > 4 else 0.0)
        )

        # Logit computation for soft probabilities
        low_logit = -score + 1.0
        med_logit = 1.0 - abs(score - 1.0)
        high_logit = score - 1.0

        logits = np.array([low_logit, med_logit, high_logit])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        prob_dict = {
            "low": float(probs[0]),
            "medium": float(probs[1]),
            "high": float(probs[2]),
        }

        top_class = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[top_class]

        return {
            "predicted_complexity": top_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "feature_dict": feat,
            "is_trained_model": False,
        }

    def train(self, feature_dicts: List[Dict[str, float]], labels: List[str]) -> Dict[str, Any]:
        """
        Train the classifier on a list of feature dicts and complexity labels.
        """
        if not feature_dicts or not labels or len(feature_dicts) != len(labels):
            raise ValueError("Training dataset must contain equal non-empty features and labels.")

        X = np.array([[f[name] for name in FEATURE_NAMES] for f in feature_dicts])
        y = np.array(labels)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)

        self.clf = clf
        self.is_trained = True
        self.save_model()

        importances = self.get_feature_importances()

        return {
            "samples_trained": len(labels),
            "classes": list(clf.classes_),
            "feature_importances": importances,
            "status": "success",
        }

    def get_feature_importances(self) -> Dict[str, float]:
        """Return dict of feature importances if trained, or default uniform weights."""
        if self.is_trained and self.clf is not None and hasattr(self.clf, "feature_importances_"):
            return {
                name: round(float(imp), 4)
                for name, imp in zip(FEATURE_NAMES, self.clf.feature_importances_)
            }
        return {name: round(1.0 / len(FEATURE_NAMES), 4) for name in FEATURE_NAMES}


# Global singleton instance
_ai_router_instance: Optional[AIRouter] = None


def get_ai_router() -> AIRouter:
    """Get or initialize the global AIRouter singleton instance."""
    global _ai_router_instance
    if _ai_router_instance is None:
        _ai_router_instance = AIRouter()
    return _ai_router_instance
