# OptiLLM — Implementation Plan

> A phase-by-phase improvement roadmap.
> Status tracked in `TASKS.md`. Last updated: 2026-09-21.

---

## Phase 1 — Critical Fixes *(Start Here)*

> Correctness bugs and production blockers. Everything downstream depends on these.

### 1.1 Fix Streaming Token Counting
- **File:** `app/services/gateway.py` — `process_stream_request()`
- **Bug:** `tokens_output=len(full_text.split())` counts words, not tokens. Every streaming cost/budget record is wrong.
- **Fix:** Replace with `count_tokens_in_string(full_text, model_to_use)` from `app/services/token_counter.py`
- **Effort:** 30 min | **Priority:** P0

### 1.2 Replace Heuristic Judge with Dual-Mode LLM-as-Judge
- **File:** `app/evaluation/judge.py`
- **Bug:** All scores (correctness, relevance, completeness, conciseness, safety) use word-overlap and regex. Scores stored in DB and surfaced in analytics are meaningless.
- **Fix:** Add `evaluate_with_llm()` async path using gpt-4o-mini as judge (sampled at `EVAL_SAMPLE_RATE`%). Add `EVAL_SAMPLE_RATE`, `EVAL_LLM_MODEL` settings. Keep heuristic as fast-path for 100% of traffic.
- **Effort:** 1 day | **Priority:** P1

### 1.3 Fix Inline Migration Hack -> Proper Alembic
- **Files:** `app/main.py` (lines 36-54), `app/db/session.py` (lines 24-49)
- **Bug:** 10+ raw `ALTER TABLE` statements in `try/except pass` at startup. Fragile, duplicated, bypasses Alembic.
- **Fix:** Generate proper Alembic migration, replace raw SQL block with `alembic upgrade head` on startup.
- **Effort:** 1 h | **Priority:** P0

### 1.4 Remove ML Artifact from Git
- **File:** `ai_router.joblib` (52KB binary in repo root)
- **Fix:** `git rm --cached ai_router.joblib`, create `scripts/train_router.py`
- **Effort:** 30 min | **Priority:** P0

### 1.5 Add Streaming Guardrails
- **File:** `app/services/gateway.py` — `process_stream_request()`
- **Bug:** No guardrail hooks in streaming path. Safety controls bypassed for all `stream=true` requests.
- **Fix:** Add identical pre-call guardrail block at top of `process_stream_request()`.
- **Effort:** 2 h | **Priority:** P1

### 1.6 Add Per-Provider Request Timeouts
- **File:** `app/providers/dispatcher.py`
- **Bug:** Single global timeout. No per-provider differentiation. Slow providers exhaust thread pool.
- **Fix:** Add `PROVIDER_TIMEOUTS` dict, pass per-provider timeout to each client.
- **Effort:** 2 h | **Priority:** P1

---

## Phase 2 — Real Differentiation

### 2.1 Online Router Learning Loop
- **New Files:** `app/engine/router_trainer.py`, `scripts/train_router.py`
- **What:** Auto-retrain AI router on production traffic via background task. A/B evaluate before hot-swap.
- **Schema:** New `router_training_labels` table.
- **Endpoint:** `POST /api/v1/router/train`
- **Effort:** 3 days | **Priority:** P1

### 2.2 Quality-Attributed Cost Analytics
- **Files:** `app/services/analytics.py`, `app/api/endpoints/analytics.py`
- **New Endpoint:** `GET /api/v1/analytics/quality-cost-tradeoff`
- **What:** Cost, quality score, quality-per-dollar per model with auto-recommendation.
- **Effort:** 1 day | **Priority:** P1

### 2.3 Per-Request User Feedback API
- **New endpoint:** `POST /api/v1/feedback/{request_id}`
- **Schema:** New `request_feedback` table: request_id, rating (1/-1/0), issue, note
- **Effort:** 4 h | **Priority:** P1

### 2.4 Smarter Context Compression
- **File:** `app/engine/compressor.py`
- **Fix:** TF-IDF sentence importance ranking using sentence-transformers; keep top-scored sentences within token budget.
- **Effort:** 2 days | **Priority:** P3

### 2.5 Prompt Template Version Control
- **File:** `app/api/endpoints/prompts.py`
- **What:** Create version, list history, compare via eval engine, rollback.
- **Effort:** 2 days | **Priority:** P3

---

## Phase 3 — Production Hardening

### 3.1 Async SQLAlchemy Sessions
- **File:** `app/db/session.py`
- **Fix:** Migrate hot async paths to AsyncSession + asyncpg.
- **Effort:** 1 day | **Priority:** P2

### 3.2 Persist FAISS Index to Disk
- **File:** `app/engine/faiss_store.py`
- **Fix:** `faiss.write_index()` on insert, `faiss.read_index()` on startup.
- **Effort:** 4 h | **Priority:** P2

### 3.3 Structured Health Checks
- **File:** `app/api/endpoints/health.py`
- **Fix:** `GET /health/ready` returns per-subsystem status (DB, Redis, FAISS, provider circuits).
- **Effort:** 4 h | **Priority:** P2

### 3.4 Degraded Mode Integration Tests
- **New file:** `tests/integration/test_degraded_mode.py`
- **Effort:** 1 day | **Priority:** P2

### 3.5 Input Size Limits
- **Files:** `app/api/endpoints/proxy.py`, `app/api/endpoints/completions.py`
- **Fix:** Body size limit + message count limit to prevent OOM.
- **Effort:** 2 h | **Priority:** P2

---

## Phase 4 — Developer Experience

### 4.1 Python SDK Completion
- **Directory:** `optillm_client/`
- **Target:** Drop-in OpenAI replacement; publish as `optillm-client` on PyPI.
- **Effort:** 3 days | **Priority:** P2

### 4.2 Complete README + Docs
- **Files:** README.md, docs/quickstart.md, docs/routing.md, docs/caching.md, docs/evaluation.md, docs/deployment.md
- **Effort:** 2 days | **Priority:** P2

### 4.3 Benchmark Suite Expansion
- **New files:** bench_cache.py, bench_router.py, bench_compression.py, bench_vs_litellm.py, generate_report.py
- **Effort:** 2 days | **Priority:** P2

### 4.4 CLI Tool
- **New file:** `optillm_client/cli.py`
- **Commands:** chat, explain, analytics, keys, cache
- **Effort:** 2 days | **Priority:** P3

### 4.5 .env.example Reorganization
- **Fix:** Group into sections with inline comments.
- **Effort:** 30 min | **Priority:** P3

---

## Phase 5 — Security

### 5.1 Replace Regex PII with Presidio
- **File:** `app/security/pii.py`
- **Fix:** Integrate presidio-analyzer + presidio-anonymizer (Microsoft, MIT).
- **Effort:** 4 h | **Priority:** P2

### 5.2 API Key Rotation & Expiry
- **Add:** `POST /api/v1/keys/{id}/rotate`, expires_at column, key scopes.
- **Effort:** 4 h | **Priority:** P3

### 5.3 Secrets Management Backend
- **New file:** `app/core/secrets_loader.py`
- **Add:** `SECRETS_BACKEND=env|aws|vault` config.
- **Effort:** 1 day | **Priority:** P3

### 5.4 Token Bucket Rate Limiter
- **File:** `app/core/rate_limiter.py`
- **Add:** TokenBucketRateLimiter class.
- **Effort:** 4 h | **Priority:** P4

---

## Phase 6 — Observability

### 6.1 Structured JSON Logging
- **File:** `app/core/logging.py`
- **Fix:** python-json-logger; parseable by Datadog, CloudWatch, Loki.
- **Effort:** 2 h | **Priority:** P2

### 6.2 Complete OpenTelemetry Tracing
- **File:** `app/core/tracing.py`
- **Spans:** cache.check, compress, route, provider.call, cache.insert
- **Effort:** 4 h | **Priority:** P2

### 6.3 Cost Attribution Dashboard Widgets
- **File:** `app/static/admin.html`
- **Add:** Cost by team/model charts, savings waterfall, quality vs cost scatter, latency by provider.
- **Effort:** 2 days | **Priority:** P3

### 6.4 Prometheus Alerting Rules
- **New file:** `deploy/prometheus/alerts.yml`
- **Alerts:** high error rate, low cache hit rate, circuit open, budget near limit.
- **Effort:** 4 h | **Priority:** P3

---

## Phase 7 — Community & Positioning

### 7.1 Public Benchmarks Page
- **Goal:** Published: gateway overhead ms, cache hit rate, router accuracy, savings %.
- **Effort:** 2 days | **Priority:** P2

### 7.2 GitHub Actions CI/CD
- **New files:** .github/workflows/ci.yml, docker.yml, benchmarks.yml
- **Effort:** 4 h | **Priority:** P2

### 7.3 OpenAI Compatibility Audit
- **Audit:** completions, embeddings, models, streaming SSE, error format, function calling, vision.
- **Effort:** 1 day | **Priority:** P4

### 7.4 Plugin System
- **New:** `app/plugins/` with OptiLLMPlugin base class + example plugins.
- **Effort:** 3 days | **Priority:** P4

---

## Priority Matrix

| Item | Effort | Priority |
|---|---|---|
| 1.1 Streaming token counting | 30 min | **P0** |
| 1.3 Alembic migrations | 1 h | **P0** |
| 1.4 Remove joblib from git | 30 min | **P0** |
| 1.2 LLM-as-judge | 1 day | **P1** |
| 1.5 Streaming guardrails | 2 h | **P1** |
| 1.6 Per-provider timeouts | 2 h | **P1** |
| 2.1 Online router learning | 3 days | **P1** |
| 2.2 Quality-cost analytics | 1 day | **P1** |
| 2.3 Feedback API | 4 h | **P1** |
| 3.1–3.5 Prod hardening | varies | **P2** |
| 4.1–4.3 Dev experience | varies | **P2** |
| 5.1 Presidio PII | 4 h | **P2** |
| 6.1–6.2 Observability | varies | **P2** |
| 7.1–7.2 Community | varies | **P2** |
| 2.4–2.5 Differentiation extras | varies | **P3** |
| 4.4–4.5, 5.2–5.3, 6.3–6.4 | varies | **P3** |
| 5.4, 7.3–7.4 | varies | **P4** |
