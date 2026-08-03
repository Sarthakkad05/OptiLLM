"""
Unit tests for app/engine/ai_router.py — Phase 10 Intelligent AI Routing
"""

import os
import pytest

from app.engine.ai_router import (
    AIRouter,
    FEATURE_NAMES,
    extract_feature_vector,
    extract_features,
    get_ai_router,
)
from app.services.router_exporter import generate_synthetic_training_dataset


def test_extract_features():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    features = extract_features(messages, model="gpt-4o")

    for name in FEATURE_NAMES:
        assert name in features
        assert isinstance(features[name], float)

    assert features["token_count"] > 0
    assert features["char_count"] > 0
    assert features["simple_keyword_count"] >= 1.0


def test_extract_feature_vector():
    messages = [{"role": "user", "content": "Write code: def hello(): pass"}]
    vector = extract_feature_vector(messages, model="gpt-4o")

    assert len(vector) == len(FEATURE_NAMES)
    assert all(isinstance(val, float) for val in vector)


def test_ai_router_predict_untrained_heuristic():

    router = AIRouter(model_path="tmp_test_ai_router.joblib")
    messages = [{"role": "user", "content": "Who is Albert Einstein?"}]

    res = router.predict(messages, model="gpt-4o")

    assert "predicted_complexity" in res
    assert res["predicted_complexity"] in ["low", "medium", "high"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert set(res["probabilities"].keys()) == {"low", "medium", "high"}
    assert res["is_trained_model"] is False


def test_ai_router_train_and_save(tmp_path):
    model_file = str(tmp_path / "ai_router.joblib")
    router = AIRouter(model_path=model_file)

    feats, labels = generate_synthetic_training_dataset()
    train_res = router.train(feats, labels)

    assert train_res["status"] == "success"
    assert train_res["samples_trained"] == len(labels)
    assert router.is_trained is True
    assert os.path.exists(model_file)

    # Test loaded model prediction
    new_router = AIRouter(model_path=model_file)
    assert new_router.is_trained is True

    messages = [
        {
            "role": "user",
            "content": "Implement an async distributed lock engine in Python with redis.",
        }
    ]
    pred = new_router.predict(messages, model="gpt-4o")

    assert pred["is_trained_model"] is True
    assert pred["predicted_complexity"] in ["low", "medium", "high"]
    assert pred["confidence"] > 0.0
