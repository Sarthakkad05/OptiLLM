"""
Router Trainer — Online Learning Loop for the AI Router.
=========================================================
Automatically retrains the RandomForest classifier on production traffic,
A/B evaluates the new model against the current one, and hot-swaps if
accuracy improves.

This is OptiLLM's key differentiator: the router learns from YOUR workload,
not a generic training set. A medical team's router will look very different
from a code-generation team's router after a week of production traffic.

Pipeline:
  1. Collect labeled training examples from request_logs + router_training_labels
  2. Train a new RandomForest classifier
  3. Evaluate on a held-out split (20%)
  4. Hot-swap if new accuracy >= current model accuracy (or no current model)
  5. Persist the new model to disk
  6. Record training run in router_training_runs table

Background task: runs every ROUTER_RETRAIN_INTERVAL_REQUESTS requests,
or can be triggered manually via POST /api/v1/router/train.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import RequestLog, RouterTrainingLabel, RouterTrainingRun
from app.engine.ai_router import FEATURE_NAMES, AIRouter, extract_features, get_ai_router
from app.core.config import settings

logger = logging.getLogger("optillm.engine.router_trainer")


# ── Label Inference ────────────────────────────────────────────────────────────

def _infer_complexity_label(log: RequestLog) -> Optional[str]:
    """
    Infer a complexity label from a request log record.
    Priority order:
      1. Explicit label from router_training_labels (human feedback)
      2. Model-routing heuristic (what model was actually needed)
    """
    # Use quality-informed routing: if quality was poor on cheap model, it was too complex
    low_complexity_models = {"gpt-4o-mini", "gemini-2.0-flash", "claude-3-5-haiku-20241022", "groq/llama3-8b-8192"}
    high_complexity_models = {"gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "gpt-4"}

    model_used = (log.model_used or "").lower()

    if any(m in model_used for m in ["mini", "flash", "haiku", "3.5-haiku", "8b"]):
        # Low quality on a cheap model → was mis-routed as low when it should be higher
        if log.quality_score and log.quality_score < 0.70:
            return "medium"
        return "low"
    elif any(m in model_used for m in ["gpt-4o", "claude-3-5-sonnet", "opus", "gemini-1.5-pro"]):
        return "high"
    else:
        return "medium"


def build_training_dataset(db: Session, limit: int = 5000) -> Tuple[List[Dict], List[str]]:
    """
    Builds training (feature_dicts, labels) from:
      1. Explicit labels in router_training_labels table (high priority)
      2. Inferred labels from request_logs (fallback)

    Returns (feature_dicts, labels) — equal-length lists.
    """
    feature_dicts = []
    labels = []
    seen_request_ids = set()

    # Priority 1: explicit human-labeled examples
    try:
        explicit_labels = (
            db.query(RouterTrainingLabel)
            .order_by(RouterTrainingLabel.created_at.desc())
            .limit(limit)
            .all()
        )
        for label_record in explicit_labels:
            messages = [{"role": "user", "content": label_record.prompt_snippet or ""}]
            feats = extract_features(messages, label_record.model_requested or "gpt-4o")
            feature_dicts.append(feats)
            labels.append(label_record.complexity_label)
            if label_record.request_log_id:
                seen_request_ids.add(label_record.request_log_id)
        logger.info("Loaded %d explicit training labels.", len(labels))
    except Exception as e:
        logger.debug("router_training_labels table not available yet: %s", e)

    # Priority 2: inferred labels from request_logs
    remaining = max(0, limit - len(labels))
    if remaining > 0:
        logs = (
            db.query(RequestLog)
            .filter(RequestLog.prompt_snippet.isnot(None))
            .filter(RequestLog.id.notin_(seen_request_ids))
            .order_by(RequestLog.timestamp.desc())
            .limit(remaining)
            .all()
        )
        for log in logs:
            label = _infer_complexity_label(log)
            if label:
                messages = [{"role": "user", "content": log.prompt_snippet or ""}]
                feats = extract_features(messages, log.model_requested or "gpt-4o")
                feature_dicts.append(feats)
                labels.append(label)

        logger.info("Loaded %d inferred labels from request_logs.", len(logs))

    return feature_dicts, labels


# ── Training & Evaluation ──────────────────────────────────────────────────────

def _evaluate_model_accuracy(
    router: AIRouter,
    feature_dicts: List[Dict],
    labels: List[str],
) -> float:
    """Evaluate accuracy of router on given (features, labels) pairs."""
    if not feature_dicts:
        return 0.0
    correct = 0
    for feats, true_label in zip(feature_dicts, labels):
        prediction = router.predict([{"role": "user", "content": ""}], "gpt-4o")
        # Use direct feature prediction to avoid re-extracting features
        import numpy as np
        feat_vec = [feats[name] for name in FEATURE_NAMES]
        if router.is_trained and router.clf is not None:
            try:
                probs = router.clf.predict_proba([feat_vec])[0]
                predicted = list(router.clf.classes_)[int(probs.argmax())]
                if predicted == true_label:
                    correct += 1
            except Exception:
                pass
    return correct / len(labels) if labels else 0.0


def train_router(
    db: Session,
    force: bool = False,
    min_samples: int = 30,
) -> Dict[str, Any]:
    """
    Full training pipeline:
      1. Build dataset from DB
      2. Split 80/20 train/eval
      3. Train new model
      4. Evaluate vs current model
      5. Hot-swap if better
      6. Record run

    Returns a dict with training results.
    """
    start_time = time.time()
    logger.info("Starting router training run...")

    feature_dicts, labels = build_training_dataset(db)
    total = len(labels)

    if total < min_samples and not force:
        msg = f"Not enough training data ({total} samples, need {min_samples}). Use force=True to override."
        logger.warning(msg)
        return {"status": "skipped", "reason": msg, "samples": total}

    if total < min_samples:
        logger.warning("Training with only %d samples (force=True).", total)

    # 80/20 train/eval split
    split_idx = int(total * 0.8)
    train_features = feature_dicts[:split_idx]
    train_labels = labels[:split_idx]
    eval_features = feature_dicts[split_idx:]
    eval_labels = labels[split_idx:]

    # Evaluate current model before training
    current_router = get_ai_router()
    current_accuracy = _evaluate_model_accuracy(current_router, eval_features, eval_labels) if eval_features else 0.0

    # Train new model
    new_router = AIRouter(model_path=settings.AI_ROUTER_MODEL_PATH + ".candidate")
    new_router.clf = None
    new_router.is_trained = False

    try:
        train_result = new_router.train(train_features, train_labels)
    except Exception as e:
        logger.error("Training failed: %s", e)
        return {"status": "error", "reason": str(e), "samples": total}

    # Evaluate new model
    new_accuracy = _evaluate_model_accuracy(new_router, eval_features, eval_labels) if eval_features else 1.0

    elapsed_ms = int((time.time() - start_time) * 1000)
    label_distribution = {
        lbl: labels.count(lbl) for lbl in ["low", "medium", "high"]
    }

    # Hot-swap decision
    swapped = False
    if new_accuracy >= current_accuracy or not current_router.is_trained:
        # Promote candidate → main model path
        import os, shutil
        candidate_path = settings.AI_ROUTER_MODEL_PATH + ".candidate"
        if os.path.exists(candidate_path):
            shutil.move(candidate_path, settings.AI_ROUTER_MODEL_PATH)

        # Reload the global singleton
        current_router.model_path = settings.AI_ROUTER_MODEL_PATH
        current_router.load_model()
        swapped = True
        logger.info(
            "Router hot-swapped! Accuracy: %.3f → %.3f (eval on %d samples)",
            current_accuracy, new_accuracy, len(eval_labels),
        )
    else:
        logger.info(
            "New model (%.3f) did not improve over current (%.3f) — keeping current.",
            new_accuracy, current_accuracy,
        )
        # Clean up candidate
        import os
        candidate = settings.AI_ROUTER_MODEL_PATH + ".candidate"
        if os.path.exists(candidate):
            os.remove(candidate)

    result = {
        "status": "success",
        "samples_trained": len(train_labels),
        "samples_evaluated": len(eval_labels),
        "current_model_accuracy": round(current_accuracy, 4),
        "new_model_accuracy": round(new_accuracy, 4),
        "model_swapped": swapped,
        "label_distribution": label_distribution,
        "feature_importances": train_result.get("feature_importances", {}),
        "training_time_ms": elapsed_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Record training run in DB
    try:
        run = RouterTrainingRun(
            samples_trained=len(train_labels),
            samples_evaluated=len(eval_labels),
            current_accuracy=current_accuracy,
            new_accuracy=new_accuracy,
            model_swapped=swapped,
            training_time_ms=elapsed_ms,
        )
        db.add(run)
        db.commit()
    except Exception as e:
        logger.debug("Could not record training run: %s", e)

    return result


# ── Background Auto-Trainer ────────────────────────────────────────────────────

_request_count_since_last_train = 0
_training_lock = asyncio.Lock()


async def maybe_trigger_auto_retrain(db: Session) -> None:
    """
    Called after each request. Triggers retraining when enough new requests
    have accumulated since the last training run.

    Controlled by ROUTER_RETRAIN_INTERVAL_REQUESTS setting.
    """
    global _request_count_since_last_train

    if not settings.ROUTER_AUTO_RETRAIN:
        return

    _request_count_since_last_train += 1

    if _request_count_since_last_train < settings.ROUTER_RETRAIN_INTERVAL_REQUESTS:
        return

    if _training_lock.locked():
        return  # Training already in progress — skip this trigger

    async with _training_lock:
        _request_count_since_last_train = 0
        logger.info(
            "Auto-retrain triggered after %d requests.",
            settings.ROUTER_RETRAIN_INTERVAL_REQUESTS,
        )
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: train_router(db))
            logger.info("Auto-retrain complete: %s", result.get("status"))
        except Exception as e:
            logger.error("Auto-retrain failed: %s", e)
