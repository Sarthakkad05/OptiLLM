# OptiLLM Router Learning-Loop Validation Report

> **Generated:** 2026-09-22 16:30 UTC
> **Seeded examples:** 140 hand-labeled prompts (`benchmarks/router_training_dataset.py`)
> **Model existed before this run:** True

This validates `app/engine/router_trainer.py` — the retrain-evaluate-hotswap pipeline described
as OptiLLM's key differentiator — by actually running it, in two ways: a clean evaluation
isolated from any DB state, and the real end-to-end pipeline as a normal user would trigger it.

## 1. Clean Evaluation (hand-labeled data only, no DB, no contamination)

Trained and evaluated entirely in-memory on an 80/20 split of just this dataset — no
`request_logs` involved — to get a trustworthy read on whether the classifier learns real
signal from clean labels.

| Metric | Value |
|---|---|
| Train / Eval size | 112 / 28 |
| Eval label distribution | `{'low': 11, 'medium': 13, 'high': 4}` |
| **Accuracy** | **0.8571** |

### Feature Importances (clean dataset only)

| Feature | Importance |
|---|---|
| `char_count` | 0.3799 |
| `token_count` | 0.2726 |
| `avg_tokens_per_turn` | 0.2586 |
| `simple_keyword_count` | 0.0497 |
| `complex_keyword_count` | 0.0364 |
| `code_signal_count` | 0.0021 |
| `turn_count` | 0.0005 |
| `user_msg_count` | 0.0002 |
| `is_expensive_model` | 0.0000 |

## 2. Real Pipeline Run (`train_router()`, as triggered via background auto-retrain or the API)

| Metric | Value |
|---|---|
| Samples trained | 193 |
| Samples evaluated (held out) | 49 |
| Accuracy before (current live model) | 0.8571 |
| Accuracy after (candidate model) | 0.7959 |
| **Model hot-swapped?** | **False** |
| Training time | 317 ms |

Note: `build_training_dataset()` pulled 102 additional inferred labels from leftover `request_logs` rows (mock/test traffic, not real quality-checked outcomes) on top of the 140 labels this script seeded — see the discovered issue below for why that matters more than it sounds.

Label distribution in the full dataset (before the 80/20 split): `{'low': 140, 'medium': 54, 'high': 48}`

### Feature Importances (real pipeline, mixed dataset)

| Feature | Importance |
|---|---|
| `char_count` | 0.5170 |
| `token_count` | 0.1990 |
| `avg_tokens_per_turn` | 0.1909 |
| `complex_keyword_count` | 0.0529 |
| `simple_keyword_count` | 0.0398 |
| `code_signal_count` | 0.0004 |
| `turn_count` | 0.0000 |
| `user_msg_count` | 0.0000 |
| `is_expensive_model` | 0.0000 |

**Hot-swap gate:** The candidate did NOT beat the current model, so the live model was left unchanged — the safety gate did its job.


## Discovered Issue: the real pipeline's eval set wasn't what we thought

`build_training_dataset()` in `router_trainer.py` builds its list as `[explicit RouterTrainingLabel
rows (newest first)] + [inferred RequestLog rows (newest first)]`, and `train_router()` then takes a
**plain positional 80/20 slice** of that concatenated list — no shuffling, no stratification by
label OR by label source. In this run, that meant: because there were more pre-existing inferred
`request_logs` rows than 20% of the total dataset, the eval slice (the last 20%) landed entirely in
the inferred-label region, and **all 140 of this script's hand-labeled examples ended up in
training, none in evaluation.** So the "Accuracy before/after" numbers in section 2 measure the
model against noisy, DB-inferred labels (derived from mock-mode test traffic, not real judged
outcomes) — not against the clean ground truth in section 1.

This is a real, reproducible pipeline behavior, not a one-off fluke: any deployment where explicit
human/feedback labels are a minority of the combined dataset will have this same property, silently,
every time it retrains. Section 1's clean accuracy is the number to trust for "does the classifier
learn anything real"; section 2 is the number to trust for "does the actual shipped pipeline behave
safely" (its hot-swap gate is still real and still fired correctly on whatever it was evaluated against).

## Interpretation

- The clean evaluation shows the classifier does learn genuine signal from the 9 structural
  features it has access to (see `app/engine/ai_router.py::FEATURE_NAMES`) — but those features are
  length/keyword/code-block signals with no semantic understanding, which caps how well this can
  ever do on prompts that are complex but short and keyword-free (e.g. a terse but deep math
  question would look "low complexity" to every one of these features).
- The hot-swap safety gate ("swap only if new_accuracy >= current_accuracy") is real code that ran
  and made a real decision in this test, not a documentation-only claim — see section 2.

## Known Limitations

- The train/eval split issue above is a genuine limitation of the shipped pipeline, not just of
  this test script.
- The hand-labeled dataset (`benchmarks/router_training_dataset.py`) reflects one person's judgment
  of complexity, not independently verified or drawn from real user traffic.
- `_evaluate_model_accuracy()` in `router_trainer.py` calls `router.predict(...)` once per sample
  with a placeholder empty-content message purely for a side effect that's never used (the return
  value is discarded and accuracy is computed separately from the real feature vector) — harmless
  to correctness, but dead code worth cleaning up.
