"""
OptiLLM AI Router Training Script
===================================
Trains (or retrains) the RandomForest complexity classifier used by the
AI Router to predict prompt complexity (low / medium / high).

Usage:
    python scripts/train_router.py                    # Use synthetic dataset
    python scripts/train_router.py --from-db          # Train on production logs
    python scripts/train_router.py --output path.joblib

The trained model is saved to AI_ROUTER_MODEL_PATH (default: ai_router.joblib).
DO NOT commit the .joblib file to git — it is listed in .gitignore.
"""

import argparse
import logging
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_router")


# ── Synthetic Training Dataset ────────────────────────────────────────────────

SYNTHETIC_DATASET = [
    # LOW complexity
    ({"role": "user", "content": "What is the capital of France?"}, "low"),
    ({"role": "user", "content": "What is 2 + 2?"}, "low"),
    ({"role": "user", "content": "Define photosynthesis."}, "low"),
    ({"role": "user", "content": "Who is the president of the United States?"}, "low"),
    ({"role": "user", "content": "Translate 'hello' to Spanish."}, "low"),
    ({"role": "user", "content": "What is the boiling point of water?"}, "low"),
    ({"role": "user", "content": "List the planets in the solar system."}, "low"),
    ({"role": "user", "content": "What year did World War II end?"}, "low"),
    ({"role": "user", "content": "True or false: the sun is a star."}, "low"),
    ({"role": "user", "content": "Summarize this paragraph in one sentence: ..."}, "low"),

    # MEDIUM complexity
    ({"role": "user", "content": "Explain how neural networks work."}, "medium"),
    ({"role": "user", "content": "Compare REST and GraphQL APIs."}, "medium"),
    ({"role": "user", "content": "Analyze the pros and cons of microservices."}, "medium"),
    ({"role": "user", "content": "Explain the difference between supervised and unsupervised learning."}, "medium"),
    ({"role": "user", "content": "How does HTTPS work? Explain TLS handshake."}, "medium"),
    ({"role": "user", "content": "What are the SOLID principles? Give examples."}, "medium"),
    ({"role": "user", "content": "Explain how Git rebasing works vs merging."}, "medium"),
    ({"role": "user", "content": "Compare Python, JavaScript, and Go for backend development."}, "medium"),

    # HIGH complexity
    ({"role": "user", "content": "Implement a binary search tree with insert, delete, and search in Python."}, "high"),
    ({"role": "user", "content": "Write a React component that fetches paginated data from an API with error handling."}, "high"),
    ({"role": "user", "content": "Design a distributed rate limiter using Redis that works across multiple servers."}, "high"),
    ({"role": "user", "content": "Refactor this code to use the repository pattern and add unit tests."}, "high"),
    ({"role": "user", "content": "Architect a real-time chat system that handles 1M concurrent users."}, "high"),
    ({"role": "user", "content": "Debug this async Python code that has a race condition: ..."}, "high"),
    ({"role": "user", "content": "Write a SQL query to find the top 10 customers by revenue in the last 90 days with proper indexing."}, "high"),
    ({"role": "user", "content": "Implement a LRU cache in Python with O(1) get and put operations using step by step reasoning."}, "high"),
    ({"role": "user", "content": "Create a system design for a URL shortener that handles 100k requests/second."}, "high"),
    ({"role": "user", "content": "Prove why quicksort has O(n log n) average time complexity and derive the recurrence."}, "high"),
]


def build_dataset_from_synthetic():
    """Build feature dicts and labels from the synthetic dataset."""
    from app.engine.ai_router import extract_features

    feature_dicts = []
    labels = []
    for msg, label in SYNTHETIC_DATASET:
        messages = [msg]
        features = extract_features(messages, "gpt-4o")
        feature_dicts.append(features)
        labels.append(label)
    return feature_dicts, labels


def build_dataset_from_db():
    """Build training data from production request_logs (requires DB)."""
    from app.db.session import SessionLocal
    from app.db.models import RequestLog
    from app.engine.ai_router import extract_features

    logger.info("Connecting to database to fetch training data...")
    db = SessionLocal()
    try:
        # Use routing decisions as ground truth labels
        logs = db.query(RequestLog).filter(
            RequestLog.routed.is_(True)
        ).limit(5000).all()

        if not logs:
            logger.warning("No routed requests found in DB. Using synthetic dataset.")
            return build_dataset_from_synthetic()

        feature_dicts = []
        labels = []
        for log in logs:
            if not log.prompt_snippet:
                continue
            messages = [{"role": "user", "content": log.prompt_snippet}]
            features = extract_features(messages, log.model_requested or "gpt-4o")
            feature_dicts.append(features)
            # Infer label from which model was actually used
            if log.model_used in {"gemini-2.0-flash", "gpt-3.5-turbo", "gpt-4o-mini"}:
                labels.append("low")
            elif log.model_used in {"gpt-4o-mini"}:
                labels.append("medium")
            else:
                labels.append("high")

        logger.info("Loaded %d training examples from database.", len(labels))
        return feature_dicts, labels
    finally:
        db.close()


def train(from_db: bool = False, output_path: str = None):
    from app.engine.ai_router import get_ai_router
    from app.core.config import settings

    output_path = output_path or settings.AI_ROUTER_MODEL_PATH

    if from_db:
        feature_dicts, labels = build_dataset_from_db()
    else:
        feature_dicts, labels = build_dataset_from_synthetic()

    logger.info("Training AI Router on %d examples...", len(labels))

    router = get_ai_router()
    router.model_path = output_path
    result = router.train(feature_dicts, labels)

    logger.info("Training complete!")
    logger.info("  Samples: %d", result["samples_trained"])
    logger.info("  Classes: %s", result["classes"])
    logger.info("  Model saved to: %s", output_path)
    logger.info("  Feature importances:")
    for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1]):
        logger.info("    %-30s %.4f", feat, imp)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OptiLLM AI Router")
    parser.add_argument("--from-db", action="store_true",
                        help="Train on production data from DB instead of synthetic dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for trained model (default: AI_ROUTER_MODEL_PATH from config)")
    args = parser.parse_args()

    train(from_db=args.from_db, output_path=args.output)
