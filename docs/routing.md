# Intelligent Routing & Online Learning

OptiLLM's routing engine automatically classifies query complexity to route simple prompts to fast, cost-effective models (e.g., `gpt-4o-mini`, `gemini-2.0-flash`, `claude-3-5-haiku`) while reserving expensive frontier models (e.g., `gpt-4o`, `claude-3-5-sonnet`) for prompts requiring deep reasoning or complex code synthesis.

> **Validation status:** the mechanism described below has been run end-to-end against real (hand-labeled) data and a real HTTP call to the training endpoint — not just asserted. See [§4](#4-does-it-actually-work-real-validation-results) for what was actually measured, including a real bug this validation found and fixed.

---

## 1. Routing Modes

Configured via `ROUTING_MODE` in `.env`:

| Mode | Behavior | Use Case |
|---|---|---|
| `rule_based` | Evaluates prompt using heuristic rules and pattern matching | Zero-ML overhead |
| `ai` | Evaluates prompt using trained RandomForest classifier (`ai_router.joblib`) | Maximum accuracy |
| `shadow` (Default) | Runs rule-based router for execution, runs AI router concurrently, logs disagreements to DB | Production observation & data gathering |

Shadow mode never changes what model is used — it only records when the AI router *would have* decided differently, as evidence for whether promoting it to `ai` mode is a good idea. That disagreement data is now actually queryable (it wasn't, until this validation pass — see §4.3).

---

## 2. Complexity Tiers

Each query is classified into one of three complexity tiers using 9 structural features — token/char count, turn count, complex/simple keyword counts, code-block signals, and whether the requested model is already an expensive one (see `app/engine/ai_router.py::FEATURE_NAMES`). These are shallow signals, not semantic understanding: a terse but genuinely hard question (e.g. a one-line math proof request) can look "low complexity" to every one of these features. Keep that in mind when trusting a `low` classification for adversarially short prompts.

1. **`low`**: Factual questions, translations, definitions, simple arithmetic, basic summaries.
   - *Target Models:* `gpt-4o-mini`, `gemini-2.0-flash`, `mistral-small`
2. **`medium`**: Multi-paragraph writing, standard code refactoring, data analysis, business logic.
   - *Target Models:* `claude-3-5-haiku`, `gpt-4o-mini`
3. **`high`**: Architectural design, Byzantine consensus, compiler parsing, mathematical proofs, security audits.
   - *Target Models:* Preserves user's requested model (e.g. `gpt-4o`, `claude-3-5-sonnet`)

---

## 3. The Online Router Learning Loop

Unlike static routers, OptiLLM is designed to continuously improve its routing model based on real production traffic, via `app/engine/router_trainer.py::train_router()`:

```
Incoming Requests ──► request_logs (inferred labels, lower quality)
                              │
User Feedback API ──► RouterTrainingLabel (explicit labels, higher quality)
                              │
Every 1000 Requests, or POST /api/v1/router/train ─► [Retrain Pipeline]
                              │
                              ├── 1. Build dataset from explicit + inferred labels
                              ├── 2. 80/20 train/eval split
                              ├── 3. Train a candidate RandomForest
                              ├── 4. Evaluate candidate vs. current model on the held-out split
                              └── 5. Hot-swap ONLY if candidate accuracy >= current accuracy
```

### Auto-Retraining Configuration

In `.env`:
```ini
ROUTER_AUTO_RETRAIN=true                 # Enable automatic background retraining
ROUTER_RETRAIN_INTERVAL_REQUESTS=1000    # Retrain after this many requests
ROUTER_MIN_TRAINING_SAMPLES=30           # Minimum labeled samples to trigger training
```

### Manual Trigger & Inspection

```bash
# Trigger a training run (a JSON body is required, even if empty — an empty
# body with no Content-Type will 422)
curl -X POST http://localhost:8000/api/v1/router/train \
  -H "Authorization: Bearer sk-optillm-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"force": true, "min_samples": 30}'

# View training history & accuracy progression
curl http://localhost:8000/api/v1/router/training-history \
  -H "Authorization: Bearer sk-optillm-dev-key"

# View recent shadow-mode disagreements (rule-based vs AI router)
curl "http://localhost:8000/api/v1/routing/shadow-disagreements?limit=20" \
  -H "Authorization: Bearer sk-optillm-dev-key"
```

Training runs in the background; check `/router/training-history` a moment later for the result.

---

## 4. Does It Actually Work? Real Validation Results

The sections above describe the intended design. This section is what was actually run and found — see `benchmarks/bench_router_learning.py`, `benchmarks/router_training_dataset.py`, and `benchmarks/ROUTER_LEARNING_REPORT.md` for the full detail and reproducer.

### 4.1 Clean accuracy on hand-labeled data

A 140-example hand-labeled dataset (50 low / 50 medium / 40 high, spanning factual Q&A, code tasks, system design, proofs, security analysis, and a few multi-turn conversations) was split 80/20 and evaluated in isolation from any database state, to get a trustworthy read on whether the classifier learns real signal.

**Result: 85.7% accuracy** on 28 held-out examples. Feature importances showed `char_count` (0.38), `token_count` (0.27), and `avg_tokens_per_turn` (0.26) dominate — i.e. the classifier is mostly learning "how long is this," which tracks with the limitation noted in §2.

### 4.2 A real bug: the shipped pipeline's eval set wasn't what it looked like

Running the *actual* `train_router()` pipeline (not the isolated test above) against a database that had both the 140 hand-labeled rows and 102 pre-existing inferred rows from incidental test traffic revealed that `build_training_dataset()` concatenates `[explicit labels] + [inferred labels]` and then takes a **plain positional 80/20 slice** — no shuffling, no stratification by label or by label source. Because there were more inferred rows than 20% of the total, the eval slice landed entirely in the inferred-label region: **all 140 hand-labeled examples ended up in training, none in evaluation.** The reported "before/after accuracy" was silently measuring against noisy, DB-inferred labels instead of the clean ground truth.

This is a real, reproducible property of the shipped code, not a one-off — any deployment where explicit labels are a minority of the combined dataset will have this happen on every retrain, silently. It's tracked as a known limitation (see `benchmarks/ROUTER_LEARNING_REPORT.md`); the fix would be a stratified/shuffled split in `build_training_dataset()`, not yet implemented.

### 4.3 A real bug that was fixed: the hot-swap safety gate was unreachable via the API

While validating this, `POST /api/v1/router/train` was found to have **two different route handlers registered at the identical path** — one in `app/api/endpoints/router_config.py` (trained the live model directly with no eval, no hot-swap check, immediate unconditional overwrite) and one in `app/api/endpoints/feedback.py` (the real pipeline described above). FastAPI/Starlette matches routes in registration order, and the unsafe one was registered first, so it silently won every time — the safety mechanism this whole page describes was **completely unreachable through the API**. Confirmed empirically via the route table and a live request before fixing it.

This has been fixed: the naive duplicate endpoint was removed (`app/api/endpoints/router_config.py`), leaving the real evaluate-then-hotswap pipeline as the only implementation reachable at that path. `tests/integration/test_router_endpoints.py::test_train_router_endpoint` was updated to actually assert the correct pipeline's behavior (it previously asserted the *unsafe* endpoint's response shape, which is how this went unnoticed).

**Live proof the fix works:** immediately after the fix, the live model (which had been corrupted to ~16% accuracy by an earlier debugging experiment that trained it on contaminated data) was retrained via the actual HTTP endpoint:

```json
{"current_model_accuracy": 0.16, "new_model_accuracy": 0.8, "model_swapped": true}
```

The gate correctly detected the broken model and swapped in a much better one — through the real API, not a direct function call. Before the fix, calling this same endpoint would have silently overwritten the model with whatever the naive path produced, with no accuracy check at all.

### 4.4 Shadow-mode disagreements are now actually visible

`route()` was already computing `shadow_disagreement` (whether the AI router's predicted complexity differed from the rule-based one) and logging it — but nothing ever persisted it, so it was invisible outside of grepping application logs. This has been fixed: `request_logs` now has `shadow_disagreement` and `ai_predicted_complexity` columns (migration `004_shadow_router_observability`), populated by `app/services/gateway.py` on every routed and non-routed request. `GET /api/v1/routing/stats` now reports a `shadow_disagreement_rate`, `GET /api/v1/routing/shadow-disagreements` lists recent events, and the admin dashboard's Analytics tab shows both.

The first real disagreement this captured, live: a request for *"Design a distributed multi-datacenter consensus protocol with fault tolerance against Byzantine nodes"* was rule-classified as `medium` (and down-routed to a cheaper model) while the AI router correctly flagged it `high` — a concrete, reproducible argument for eventually promoting the AI router out of shadow mode, once enough of this evidence accumulates.

### 4.5 Known limitations

- The train/eval split issue in §4.2 is unfixed — trust the clean-eval methodology (§4.1) over the shipped pipeline's self-reported accuracy until it's addressed.
- The hand-labeled dataset reflects one person's judgment of complexity, not independently verified or drawn from real user traffic.
- `_evaluate_model_accuracy()` in `router_trainer.py` calls `router.predict(...)` once per sample with a placeholder empty-content message purely for a side effect that's never used — harmless, but dead code.
