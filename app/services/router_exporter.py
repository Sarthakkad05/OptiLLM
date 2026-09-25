"""
Synthetic Training Dataset Generator
Generates a small synthetic seed dataset for instantly bootstrapping the AI
Router classifier (used by tests and for cold-start training before enough
real/labeled traffic exists — see app/engine/router_trainer.py for the real
retrain-evaluate-hotswap pipeline that should be used once real data exists).
"""

import logging
from typing import Dict, List, Tuple

from app.engine.ai_router import extract_features

logger = logging.getLogger("optillm.services.router_exporter")


def generate_synthetic_training_dataset() -> Tuple[List[Dict[str, float]], List[str]]:
    """
    Generates a synthetic seed dataset covering simple, medium, and complex prompts
    for instant training and bootstrapping the ML classifier.
    """
    dataset: List[Tuple[List[Dict], str, str]] = [
        # Simple / LOW
        ([{"role": "user", "content": "What is the capital of France?"}], "gpt-4o", "low"),
        ([{"role": "user", "content": "Translate 'hello' into Spanish."}], "gpt-4o", "low"),
        ([{"role": "user", "content": "List 3 benefits of drinking water."}], "gpt-4o", "low"),
        ([{"role": "user", "content": "Define what an API is in simple terms."}], "gpt-4o", "low"),
        ([{"role": "user", "content": "Who wrote Romeo and Juliet?"}], "gpt-4o", "low"),
        ([{"role": "user", "content": "Calculate 15 * 8."}], "gpt-4o", "low"),
        ([{"role": "user", "content": "Correct the spelling: recipient."}], "gpt-4o", "low"),
        # Medium / MEDIUM
        ([{"role": "user", "content": "Summarize the key differences between REST and GraphQL APIs."}], "gpt-4o", "medium"),
        ([{"role": "user", "content": "Explain how database indexing works with a simple example."}], "gpt-4o", "medium"),
        ([{"role": "user", "content": "Format this JSON string and explain the fields."}], "gpt-4o", "medium"),
        ([{"role": "user", "content": "Compare Python list vs set performance for lookups."}], "gpt-4o", "medium"),
        ([{"role": "user", "content": "How do Docker containers isolate process resources?"}], "gpt-4o", "medium"),
        # Complex / HIGH
        (
            [
                {
                    "role": "user",
                    "content": (
                        "Implement an async deadlock-free transaction runner in Python with SQLAlchemy "
                        "and write unit tests verifying retry backoff behavior. Include code blocks and class structures."
                    ),
                }
            ],
            "gpt-4o",
            "high",
        ),
        (
            [
                {
                    "role": "user",
                    "content": (
                        "Architect a high-availability distributed microservice system handling 100k RPS "
                        "with circuit breakers, redis caching, and multi-region failover. Step by step design."
                    ),
                }
            ],
            "gpt-4o",
            "high",
        ),
        (
            [
                {
                    "role": "user",
                    "content": (
                        "Debug this C++ memory leak stack trace: ```cpp void* p = malloc(1024); ``` "
                        "and refactor it into modern C++ RAII smart pointers with full safety guarantees."
                    ),
                }
            ],
            "gpt-4o",
            "high",
        ),
        (
            [
                {
                    "role": "user",
                    "content": (
                        "Prove the time complexity of QuickSort using mathematical induction and recurrence relations."
                    ),
                }
            ],
            "gpt-4o",
            "high",
        ),
    ]

    features_list: List[Dict[str, float]] = []
    labels_list: List[str] = []

    for msgs, model, label in dataset:
        feat = extract_features(msgs, model)
        features_list.append(feat)
        labels_list.append(label)

    return features_list, labels_list
