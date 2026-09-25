# OptiLLM — Comprehensive Interview Question Bank

> All questions and answers below are grounded in the **actual source code and documented design decisions** of the OptiLLM repository. Answers cite specific files and line numbers where relevant.

---

## 📌 Table of Contents

1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [System Architecture](#2-system-architecture)
3. [Semantic Caching (FAISS + Redis)](#3-semantic-caching-faiss--redis)
4. [Context Compression (TF-IDF)](#4-context-compression-tf-idf)
5. [Intelligent Routing & Complexity Classification](#5-intelligent-routing--complexity-classification)
6. [Online Learning Loop (Router Retraining)](#6-online-learning-loop-router-retraining)
7. [Provider Dispatch & Resilience](#7-provider-dispatch--resilience)
8. [Security & Rate Limiting](#8-security--rate-limiting)
9. [Observability & Monitoring](#9-observability--monitoring)
10. [Database & Migrations](#10-database--migrations)
11. [Deployment & High Availability](#11-deployment--high-availability)
12. [Testing Strategy](#12-testing-strategy)
13. [Design Decisions & Trade-offs](#13-design-decisions--trade-offs)
14. [Bugs You Found & Fixed](#14-bugs-you-found--fixed)
15. [Behavioral / HR Questions](#15-behavioral--hr-questions)

---

## 1. Project Overview & Motivation

**Q1: What is OptiLLM and what problem does it solve?**

> OptiLLM is a high-performance, drop-in AI gateway that sits transparently between client applications and LLM providers (OpenAI, Anthropic, Gemini, Groq, Mistral, Azure, Bedrock, Ollama). It executes a 4-tier optimization pipeline — Semantic Caching → Context Compression → Intelligent Routing → Feedback Loop — targeting 40–80% cost reduction without sacrificing output quality. Applications point their OpenAI SDK `base_url` at `http://localhost:8000/v1` and need zero code changes.

---

**Q2: Why build a gateway instead of calling LLM APIs directly?**

> Direct API calls have no cost optimization layer. OptiLLM adds:
> - **Semantic cache** that returns stored answers for paraphrase-equivalent queries, eliminating real API calls entirely for repeated questions
> - **Down-routing** that detects simple prompts and sends them to cheaper models (e.g., gpt-4o → gpt-4o-mini saves ~97% per token)
> - **Context compression** that reduces token count before every uncached call
> - **Provider resilience** with circuit breakers and multi-provider fallback, so no single provider outage breaks the application

---

**Q3: How does OptiLLM compare to LiteLLM or Portkey?**

> | Feature | OptiLLM | LiteLLM | Portkey |
> |---|:---:|:---:|:---:|
> | Self-Learning Router (Continuous Retraining) | ✅ Unique | ❌ Static Rules | ❌ Static Rules |
> | TF-IDF Informational Context Compression | ✅ Unique | ❌ | ❌ |
> | Quality-Attributed Cost Analytics | ✅ Unique | ❌ Spend only | ❌ Spend only |
> | Semantic Vector Cache | ✅ Built-in | ⚠️ Plugin only | ✅ |
> | Dual-Mode LLM-as-Judge Evaluation | ✅ Built-in | ❌ | ⚠️ Add-on |

---

**Q4: What are the benchmark results you've validated?**

> From `docs/benchmarks.md` — all local/synthetic unless noted:
> - Semantic cache hit latency: **8.36 ms** (vs. ~1,240 ms direct API call)
> - Router classification overhead P50: **1.60 ms**
> - TF-IDF compression: **68.8% token reduction** on conversational prompts
> - **HA concurrency test**: 3,300 requests at 100 concurrent across 2 Docker replicas — 0 failures, ~50/50 load split
> - Live-traffic validation (real OpenAI calls): down-routed responses held quality (avg delta −0.0025); compression showed no meaningful quality loss on tested conversations

> ⚠️ Honest caveat: cost-savings projections (40–80%) are list-price arithmetic, not observed real-world outcomes.

---

## 2. System Architecture

**Q5: Walk me through a request's lifecycle from the client to the response.**

> ```
> Client (OpenAI SDK) → nginx (LB) → FastAPI app
>   ├─ Middleware: RequestBodyLimitMiddleware (1MB cap) + RequestLoggingMiddleware
>   ├─ Auth: API key scope check + Token Bucket rate limit (RPM/TPM)
>   ├─ 1. Semantic Cache Lookup (Redis → FAISS)
>   │      HIT → return immediately (~8 ms)
>   │      MISS ↓
>   ├─ 2. Context Compression (heuristic clean + TF-IDF truncation)
>   ├─ 3. Intelligent Router (rule-based score + AI classifier in shadow mode)
>   ├─ 4. Provider Dispatch (circuit breaker → primary → fallback → mock)
>   └─ 5. Post-Execution (cache insert, cost attribution, quality judge, feedback loop)
> ```
> Source: `app/main.py`, `app/services/gateway.py`, `docs/architecture.md`

---

**Q6: What is the tech stack?**

> - **Framework**: FastAPI (Python 3.11+) with Uvicorn ASGI
> - **ORM/DB**: SQLAlchemy + Alembic migrations; SQLite for local dev, PostgreSQL in production
> - **Vector Search**: FAISS (local in-process) + Redis for distributed vector similarity
> - **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional) via sentence-transformers
> - **ML Router**: scikit-learn RandomForest classifier (`.joblib`)
> - **Deployment**: Docker Compose (nginx + N replicas + Postgres + Redis + Prometheus + Grafana)
> - **CI**: GitHub Actions (`ci.yml`)

---

**Q7: How does the gateway stay stateless to support horizontal scaling?**

> All shared mutable state lives in external services:
> - **Request logs & analytics**: PostgreSQL (shared across replicas)
> - **Cache data**: Redis (distributed) + FAISS (rebuilt from DB on startup via `sync_cache_on_startup()`)
> - **Rate limit counters**: `HybridRateLimiter` uses Redis with in-memory fallback
> - **API keys & budgets**: PostgreSQL
>
> Replicas themselves hold no persistent state — nginx round-robins freely, confirmed in the concurrency test (50/50 split).

---

## 3. Semantic Caching (FAISS + Redis)

**Q8: How does the semantic cache work?**

> Source: `app/engine/cache.py`
>
> 1. Extract the **last user message** as the lookup text (`_extract_lookup_text`)
> 2. Generate a **384-dimensional normalized embedding** using `all-MiniLM-L6-v2`
> 3. **Tier 1 — Redis**: cosine similarity lookup across in-memory Redis vectors
> 4. **Tier 2 — FAISS fallback**: inner-product search (equivalent to cosine on normalized vectors) on the local FAISS index
> 5. If score ≥ `SIMILARITY_THRESHOLD` (default 0.85), retrieve response from DB — checking namespace isolation and TTL expiry
> 6. **HIT**: return cached response (~8 ms). **MISS**: proceed to compression → routing → provider

---

**Q9: Why use both Redis AND FAISS? Why not just one?**

> - **Redis** is distributed — shared across all replicas with low latency. But Redis vector search requires the `RediSearch` module; it gracefully degrades when unavailable.
> - **FAISS** is always available (in-process) but is per-replica and not shared. It's rebuilt from the DB on startup via `sync_cache_on_startup()` to avoid index drift.
> - Together: Redis handles the hot path across replicas; FAISS is the reliable fallback when Redis isn't configured.

---

**Q10: How do you handle cache invalidation and TTL?**

> - Every cache entry has an optional `expires_at` timestamp stored in the DB
> - `CACHE_TTL_SECONDS` (from settings) sets the default TTL; per-request TTL override is supported
> - Expiry is checked in `check_cache()` before returning any FAISS hit
> - Redis entries get native key TTL via `insert_redis_cache(..., ttl_seconds=ttl)`
> - Full clear: `clear_cache()` deletes from DB, resets the FAISS index (`faiss_store.reset_index()`), and clears Redis

---

**Q11: What is the similarity threshold and why does it matter?**

> Default: **0.85** cosine similarity. From `docs/benchmarks.md`:
>
> | Threshold | Hit Rate | Latency | Best For |
> |---|---|---|---|
> | 0.80 | 100% | 16 ms | FAQ / Customer Support |
> | **0.85** | **100%** | **7.4 ms** | **General (default)** |
> | 0.90 | 60% | 7.1 ms | Financial analysis |
> | 0.95 | 0% | 7.0 ms | Code / Math proofs |
>
> Lower threshold → more hits but risk of returning semantically wrong answers. Higher → more accuracy but lower hit rate.

---

**Q12: How do you prevent cross-tenant cache leakage in a multi-tenant setup?**

> `check_cache()` applies a `tenant_id` filter when querying the DB after a FAISS hit. Even if FAISS returns a vector match, the DB query filters by namespace:
> ```python
> entry_query = entry_query.filter(CacheEntry.tenant_id == effective_namespace)
> ```
> Tenant A's cached response will never be returned for Tenant B's semantically similar request.

---

## 4. Context Compression (TF-IDF)

**Q13: How does the compression pipeline work?**

> Source: `app/engine/compressor.py` — two-pass pipeline:
>
> **Pass 1 — Heuristic Cleaning**:
> - Collapse 3+ consecutive newlines to 2
> - Collapse 3+ spaces to 1
> - Remove repeated markdown separators (`---`, `___`)
> - Deduplicate consecutive identical lines
>
> **Pass 2 — Token-Aware TF-IDF Truncation** (if tokens > `MAX_TOKENS_THRESHOLD = 2000`):
> - Split content into sentences
> - Score each sentence using `TfidfVectorizer(stop_words='english')`
> - Apply position multipliers: first sentence ×1.5, last sentence ×1.3 (edges matter most for LLMs)
> - Greedily select highest-scoring sentences within the token budget
> - Return sentences in original chronological order (not sorted by score)

---

**Q14: Why TF-IDF rather than a simpler truncation strategy?**

> The simpler alternative is "center truncation" — keep first 30% + last 70% of text. The problem is it blindly discards middle content that may contain key definitions or context. TF-IDF ranks sentences by **informational density** relative to the rest of the conversation, so it keeps semantically important content from anywhere in the conversation.
>
> The design doc cites the "Lost in the Middle" LLM phenomenon — models perform better when key information is at edges, which is why the position multipliers boost first and last sentences.

---

**Q15: How do you ensure compression doesn't degrade answer quality?**

> 1. **Original messages are never modified** — compression operates on a `messages_to_send` copy; the original is used for cache lookup
> 2. **System messages get a protected budget** (30% of total) and are only tail-trimmed, never center-truncated
> 3. **The last user message is always preserved whole** — it's the live query; truncating it would be catastrophic
> 4. **Live validation**: `benchmarks/bench_live.py` against real OpenAI calls showed compression quality deltas of +0.01 and +0.02 on the tested conversations

---

**Q16: What are the three compression modes?**

> - **`minimal`**: Heuristic cleaning only (deduplication, whitespace) — 0.4% reduction, 0.62 ms
> - **`smart`** (default): Full TF-IDF pipeline — 68.8% reduction on conversational prompts, 5.56 ms
> - **`aggressive`**: TF-IDF with a lower token ceiling (`min(max_tokens, 1000)`) — forces harder compression

---

## 5. Intelligent Routing & Complexity Classification

**Q17: How does the rule-based router classify complexity?**

> Source: `app/engine/router.py` — four independent signals, each returns a score:
>
> | Signal | LOW (0) | MED (1) | HIGH (2) |
> |---|---|---|---|
> | `_score_by_tokens` | < 80 tokens | 80–400 | > 400 |
> | `_score_by_keywords` | simple kws (−1) | — | complex kws |
> | `_score_by_conversation_depth` | ≤ 1 turn | 2–4 turns | > 4 turns |
> | `_score_by_code_content` | no code | — | code signals (```, `def`, `class`) |
>
> Total score ≤ 0 → LOW, 1–2 → MEDIUM, ≥ 3 → HIGH. Routing table: LOW → `gemini-2.0-flash`, MEDIUM → `gpt-4o-mini`, HIGH → preserve requested model.

---

**Q18: What are the three routing modes and when would you use each?**

> - **`rule_based`**: Pure heuristic, zero ML overhead. Use when latency is critical or model isn't trained yet.
> - **`ai`**: RandomForest classifier. Uses rule-based as fallback if `confidence < AI_ROUTER_CONFIDENCE_THRESHOLD`. Use when you have enough labeled traffic and the model is well-trained.
> - **`shadow`** (default): Rule-based for execution + AI router runs in parallel. Disagreements are logged to DB but never affect routing. Use in production to gather evidence before promoting the AI router.

---

**Q19: What happens in shadow mode when the two routers disagree?**

> From `router.py`:
> ```python
> shadow_disagreement = (ai_complexity_str != rule_complexity_str)
> if shadow_disagreement:
>     logger.warning("Shadow mode disagreement: Rule=%s vs AI=%s", ...)
> ```
> The disagreement is logged, stored in `request_logs.shadow_disagreement`, and visible via `GET /api/v1/routing/shadow-disagreements`. The live system captured a real example: a request about "distributed multi-datacenter consensus protocol with Byzantine fault tolerance" was rule-classified as `medium` but AI correctly flagged `high`.

---

**Q20: Why does the router skip routing for cheap models like `gpt-4o-mini`?**

> ```python
> if requested_model in _CHEAP_MODELS:
>     return {"model_used": requested_model, "routed": False, ...}
> ```
> There's nothing to save — the model is already cost-optimized. Routing it to another cheap model adds latency with zero benefit.

---

**Q21: What's the `/router/explain` endpoint and why is it useful?**

> `POST /api/v1/router/explain` calls `explain_routing()` and returns a full explainability report:
> - Rule-based score breakdown (each signal's contribution)
> - AI router prediction + confidence + per-class probabilities
> - Which model would be selected and why
> - Whether shadow disagreement exists
> - Whether confidence fallback would trigger
>
> This lets you inspect routing decisions before they happen — valuable for debugging misclassifications (like the Byzantine consensus example above).

---

## 6. Online Learning Loop (Router Retraining)

**Q22: How does OptiLLM's router learn from production traffic?**

> Source: `app/engine/router_trainer.py`. Pipeline:
> 1. **Build dataset**: query `request_logs` (inferred labels) + `router_training_labels` (explicit human feedback, higher priority). Label inference: if a cheap model returned quality < 0.70, it was mis-routed → relabel as `medium`.
> 2. **Feature extraction**: 9 features from `FEATURE_NAMES` — token count, char count, turn count, complex/simple keyword counts, code-block presence, model tier, etc.
> 3. **Train RandomForest** on 80% of data
> 4. **Evaluate** on the 20% held-out split
> 5. **Hot-swap** only if new accuracy ≥ current model accuracy
> 6. Persist `.joblib` to disk, record training run in DB

---

**Q23: When is retraining triggered?**

> Two ways:
> 1. **Automatic**: Every `ROUTER_RETRAIN_INTERVAL_REQUESTS` requests (default 1000), if `ROUTER_AUTO_RETRAIN=true` and at least `ROUTER_MIN_TRAINING_SAMPLES=30` labeled examples exist
> 2. **Manual**: `POST /api/v1/router/train` with `{"force": true}`

---

**Q24: You mentioned finding and fixing a critical bug in the training pipeline. Describe it.**

> Source: `docs/routing.md §4.3`
>
> **Bug**: Two route handlers were registered at the identical path `POST /api/v1/router/train`:
> - One in `router_config.py`: trained the live model directly — no eval, no hot-swap check, immediate unconditional overwrite of the `.joblib` file
> - One in `feedback.py`: the real pipeline with the evaluate-then-hotswap safety gate
>
> FastAPI matches routes in registration order. The unsafe one was registered first, so **every call to the training API silently bypassed the safety gate**.
>
> **Fix**: Removed the naive duplicate endpoint from `router_config.py`, leaving only the safe pipeline reachable. Updated the test which had been asserting the wrong endpoint's response shape.
>
> **Live proof**: After the fix, triggered a retrain via HTTP. The model had been corrupted to ~16% accuracy from a debugging experiment. The gate correctly swapped in a new model at 80% accuracy: `{"current_model_accuracy": 0.16, "new_model_accuracy": 0.8, "model_swapped": true}`.

---

**Q25: What's the train/eval split bug you documented?**

> Source: `docs/routing.md §4.2`
>
> `build_training_dataset()` concatenates `[explicit labels] + [inferred labels]` and takes a plain **positional 80/20 slice** — no shuffling, no stratification. Because explicit labels are inserted first and are a minority of the total dataset, they all end up in the training portion and **none in the evaluation set**.
>
> The reported "eval accuracy" was measuring against noisy inferred labels, not the clean hand-labeled ground truth. The fix would be a stratified/shuffled split — not yet implemented (tracked as a known limitation).

---

## 7. Provider Dispatch & Resilience

**Q26: How does the circuit breaker work?**

> Source: `app/core/circuit_breaker.py`. Three states per provider:
>
> - **CLOSED** (healthy): Traffic flows normally
> - **OPEN** (tripped): Provider disabled after N consecutive failures (`CIRCUIT_BREAKER_FAILURE_THRESHOLD`)
> - **HALF_OPEN** (recovering): After `CIRCUIT_BREAKER_RECOVERY_TIME` seconds, allows one trial request. Success → CLOSED; failure → back to OPEN
>
> Latency tracking uses an exponential moving average: `latency = latency * 0.7 + new_sample * 0.3`.

---

**Q27: What happens when the primary provider fails?**

> The dispatcher tries providers in priority order. If the circuit is OPEN for the primary, it skips to the fallback provider. If all real providers fail (or no API keys are configured), it falls back to a **mock provider** that returns a realistic-looking response — so development and testing never require real API keys.

---

**Q28: How do you handle multi-provider support without breaking the OpenAI SDK interface?**

> All provider clients (`openai_client.py`, `anthropic_client.py`, `gemini_client.py`, etc.) implement the same `BaseProviderClient` abstract interface with a uniform `complete(messages, model, **kwargs) → CompletionResponse` signature. The dispatcher calls the interface; the client adapts the request to provider-specific formats internally. The gateway returns an OpenAI-compatible response object regardless of which provider actually handled the request.

---

## 8. Security & Rate Limiting

**Q29: How is rate limiting implemented?**

> Source: `app/core/rate_limiter.py`. Three layers:
>
> 1. **`TokenBucketRateLimiter`**: Per-key RPM limiting with burst tolerance (`burst_factor=1.5`). Allows short bursts above the steady-state rate.
> 2. **`GlobalRateLimiter`**: Gateway-wide cap across all clients to prevent distributed overload
> 3. **`HybridRateLimiter`**: Uses Redis for distributed state (shared across replicas) with transparent in-memory fallback if Redis is unavailable
> 4. **TPM sliding window**: Separate token-per-minute tracker alongside RPM
> 5. **`AutoBlocklist`**: Automatically temporary-bans IPs that repeatedly trigger 429s
>
> Standard `X-RateLimit-*` response headers are set on every response.

---

**Q30: How does API key authentication and scoping work?**

> Source: `app/core/auth.py`. API keys have:
> - **Scopes**: `["chat", "admin", "readonly"]` — endpoints check required scopes
> - **Per-key RPM/TPM limits** that override the global defaults
> - **Budget limits**: enforced by `budget_manager.py`, rejecting requests when a key's spend cap is hit
>
> The dev key `sk-optillm-dev-key` is always valid in development mode. `API_KEY_AUTH_ENABLED=false` disables auth entirely for local testing.

---

**Q31: How do you protect against oversized requests?**

> `RequestBodyLimitMiddleware` in `app/core/middleware.py` caps request bodies at `MAX_REQUEST_BYTES` (1MB default) and limits message arrays to 200 entries. Oversized requests are rejected with a 413 before any processing.

---

## 9. Observability & Monitoring

**Q32: What metrics does OptiLLM expose?**

> `GET /metrics` returns Prometheus-format metrics via `app/core/metrics.py`:
> - Request counts per model, provider, cache source
> - Token counts (input/output) and estimated cost per request
> - Cache hit/miss rates
> - Router decision distribution (low/medium/high)
> - Shadow disagreement rate
> - Provider latency histograms
> - Circuit breaker state per provider
>
> Grafana is provisioned with a pre-built OptiLLM dashboard at `localhost:3000`.

---

**Q33: What does the Admin Dashboard show?**

> A single-file HTML dashboard (`app/static/admin.html`, served at `/admin`):
> - **Analytics**: cost/savings breakdown, provider latency & SLA compliance, shadow-mode disagreements
> - **Playground**: send a real prompt through the pipeline and see routing decision, cache status, cost/savings live
> - **Keys management**: create/revoke API keys with scopes and budget limits
> - **Cache management**: inspect stats, warm, or clear cache
>
> All panels wire to real API endpoints — not mock data.

---

**Q34: How do you trace a specific request end-to-end?**

> Every request gets a UUID logged at entry. `app/core/tracing.py` provides structured span logging. The `RequestLog` DB record stores: request ID, model requested, model used, provider, tokens in/out, cost, quality score, cache source, routing decision, shadow disagreement flag, and latency. Query via the analytics endpoints or directly in the DB.

---

## 10. Database & Migrations

**Q35: How do you handle schema migrations across multiple replicas starting simultaneously?**

> Source: `docs/deployment.md`. Every replica runs `alembic upgrade head` at startup (`app/main.py`).
>
> The problem: two replicas starting on a fresh DB would both try to create tables simultaneously → `UniqueViolation` on `alembic_version`. Reproduced 3/3 times in testing.
>
> **Fix**: `migrations/env.py` takes a **PostgreSQL advisory lock** before migrating. Replicas serialize: one migrates while others wait, then see the DB already at `head` and continue immediately. Verified 0/3 failures after fix.

---

**Q36: How do you keep the FAISS index in sync with the database across restarts?**

> `sync_cache_on_startup(db)` runs at every startup (in `app/main.py`'s lifespan context):
> 1. Fetch all `CacheEntry` rows from DB in `faiss_index_id` order
> 2. Batch-generate embeddings for all `prompt_text` values
> 3. Call `faiss_store.rebuild_from_entries(pairs)` to reconstruct the exact index
> 4. Optionally re-populate Redis if connected
>
> This ensures FAISS ID → DB entry mapping is always consistent, even after a crash or scale event.

---

## 11. Deployment & High Availability

**Q37: Describe the Docker Compose production stack.**

> From `docker-compose.yml`:
> - **nginx**: Published on host port 8000. Round-robins across `optillm` replicas using Docker's embedded DNS resolver (`resolver 127.0.0.11 valid=5s`) — re-resolves `optillm` on every request so `--scale optillm=N` works without reconfiguring nginx.
> - **optillm (×N)**: Stateless gateway replicas (just `expose` port, not `ports`)
> - **db**: PostgreSQL 15
> - **redis**: Redis 7 (shared cache + rate-limit state)
> - **prometheus / grafana**: Metrics + pre-provisioned dashboard
> - `.env` is NOT baked into the image (`.dockerignore`) — reaches containers via `env_file`
> - Embedding model is baked in at build time → replicas start in seconds without HuggingFace access

---

**Q38: What's the HA validation result?**

> `benchmarks/bench_concurrency.py` against 2 replicas + nginx + shared Postgres/Redis (mock provider mode):
>
> | Requests | Concurrency | Failures | Throughput | Load Split |
> |---:|---:|---:|---:|---|
> | 300 | 30 | 0 | 23.9 req/s | 153 / 147 |
> | 1,000 | 50 | 0 | 25.3 req/s | 516 / 484 |
> | 2,000 | 100 | 0 | 25.0 req/s | 1032 / 968 |
> | 3,300 | 100 | 0 | ~25 req/s | ~50/50 |
>
> Zero failures, exact DB row counts (nothing dropped or double-counted), near-perfect load distribution.

---

**Q39: How do you handle Kubernetes readiness probes?**

> `GET /health/ready` returns a structured response with database connectivity status and uptime. Kubernetes uses this to know when a replica is safe to receive traffic — a replica that hasn't completed migrations yet won't pass the readiness check.

---

## 12. Testing Strategy

**Q40: What's your test coverage and structure?**

> 166 tests organized in:
> - `tests/unit/`: Pure unit tests — no external service dependencies (run with `make test-fast`)
>   - `test_security_phase5.py`: PII detection, guardrails
>   - `test_plugins.py`: Plugin lifecycle hooks
>   - `test_observability_phase6.py`: Metrics and tracing
> - `tests/integration/`: Tests that exercise the full gateway
>   - `test_openai_compatibility.py`: End-to-end OpenAI SDK compatibility
>   - `test_degraded_mode.py`: Resilience — what happens when Redis/Postgres are unavailable
>   - `test_router_endpoints.py`: Training pipeline and routing APIs
>
> CI runs on GitHub Actions (`ci.yml`) on every push.

---

**Q41: How do you test the system without real API keys?**

> The mock provider mode: when no API keys are configured (or in test environments), the dispatcher falls back to a mock client that returns a realistic-looking `CompletionResponse` with fake content. This allows the full pipeline — auth, caching, compression, routing, cost attribution — to be exercised in CI with zero external calls and zero cost.

---

## 13. Design Decisions & Trade-offs

**Q42: Why use RandomForest for the router rather than a neural network?**

> - **Sub-2ms inference**: RandomForest on 9 tabular features is orders of magnitude faster than any neural net. The router must add negligible overhead.
> - **No GPU required**: Runs on any CPU in the gateway process
> - **Interpretable**: `feature_importances_` explains which signals drive decisions (validation found char_count 0.38, token_count 0.27 dominate)
> - **Retrainable in seconds**: Full retrain on thousands of examples takes < 1 second; a neural net would take minutes
>
> The trade-off is that it can't capture semantic meaning — a terse but genuinely hard question looks "low complexity" to all 9 features. That's a documented limitation.

---

**Q43: Why embed the `all-MiniLM-L6-v2` model inside the Docker image at build time?**

> Replicas start in seconds without network access to HuggingFace. In a production scale event, new replicas come up and start serving traffic immediately rather than spending 30–60 seconds downloading a 90MB model. The trade-off is a larger image size.

---

**Q44: Why does compression operate on `messages_to_send` rather than `messages`?**

> The original `messages` list is used for the **semantic cache lookup** — it must be unmodified to get the correct embedding. If we compressed before the cache lookup, semantically equivalent queries might produce different embeddings after compression, destroying cache hit rates. Compression only applies to what gets sent to the LLM.

---

**Q45: How did you decide the 0.85 cosine similarity threshold?**

> From benchmarking across thresholds (0.80, 0.85, 0.90, 0.95):
> - 0.85 gives 100% hit rate on the synthetic paraphrase set at 7.4ms latency
> - 0.90 drops to 60% hit rate — too conservative for general use
> - 0.80 gives 100% but at 16ms latency — slower due to more candidates passing threshold checks
>
> 0.85 is the default; it's tunable per-request and per-namespace for stricter use cases (financial, code gen).

---

## 14. Bugs You Found & Fixed

**Q46: What bugs did you discover during development that were particularly tricky?**

> Three real bugs documented in `docs/routing.md §4`:
>
> 1. **Duplicate route handler (critical)**: Two handlers at the same POST path; the unsafe one won due to FastAPI registration order. The safety gate (eval before hotswap) was completely unreachable. Fixed by removing the duplicate.
>
> 2. **Train/eval split contamination**: Positional 80/20 split on an ordered dataset meant all hand-labeled (high-quality) examples landed in training, none in evaluation. The reported "eval accuracy" was measured on noisy inferred labels. Unfixed — documented as a known limitation.
>
> 3. **Shadow disagreement was invisible**: The code computed `shadow_disagreement` but never persisted it to the DB. Added `shadow_disagreement` and `ai_predicted_complexity` columns via migration `004`, wrote them in `gateway.py`, and exposed them via new API endpoints and the admin dashboard.
>
> 4. **FAISS index drift on restart**: After restarts, FAISS integer IDs no longer matched DB rows. Fixed with `sync_cache_on_startup()` that rebuilds the exact index from DB on every startup.
>
> 5. **Concurrent replica startup UniqueViolation**: Reproduced 3/3 times; fixed with PostgreSQL advisory lock in `migrations/env.py`.

---

## 15. Behavioral / HR Questions

**Q47: Why did you build OptiLLM?**

> *Suggested angle*: LLM API costs compound fast at scale. I wanted to understand the full stack of optimization — vector similarity, ML classification, online learning, resilience patterns — not as individual academic exercises but as a unified system. Building a drop-in replacement for the OpenAI API forces you to handle real engineering constraints: correctness (you can't break the API contract), latency (every millisecond of overhead erodes the value proposition), and reliability (a failed gateway is worse than no gateway).

---

**Q48: What would you do differently if you started over?**

> - The train/eval split in the router trainer would be stratified and shuffled from day one
> - I'd add integration tests that specifically detect duplicate route registration (the silent override bug)
> - I'd build the shadow-mode persistence (migration `004`) before shipping shadow mode, not as a retroactive fix
> - The nginx `large_client_header_buffers` config would be in the initial setup — the "cookie too large" failure is a common gotcha with browser-based admin UIs

---

**Q49: How did you validate that the system actually works end-to-end?**

> Multiple levels:
> 1. **Unit tests** (166 total) covering individual components
> 2. **Integration tests** (`test_openai_compatibility.py`) hitting the full gateway
> 3. **Live-traffic benchmark** (`benchmarks/bench_live.py`) against real OpenAI API with real LLM-as-judge quality scoring
> 4. **HA concurrency test** (`benchmarks/bench_concurrency.py`) against the full Docker Compose stack — 3,300 requests, 0 failures
> 5. **Manual validation** of every admin dashboard panel against a running instance (zero console errors)
> 6. **Bug reproduction**: every bug in §4 of `docs/routing.md` was reproduced deterministically before being fixed

---

**Q50: What's the most important thing you learned building this?**

> *Suggested angle*: That "it works" and "it works correctly in production" are separated by a large gap. The duplicate route handler bug is a perfect example — the feature appeared to work in tests because the tests were asserting the wrong endpoint's behavior. The fix required running the actual HTTP endpoint against a live DB and observing the safety gate trigger. Observability (shadow mode, disagreement logging, training history) isn't optional — it's what lets you know whether the system is doing what you think it's doing.

---

## 🔥 Rapid-Fire Technical Questions

| Question | Short Answer |
|---|---|
| What embedding model do you use? | `all-MiniLM-L6-v2`, 384-dimensional, normalized |
| What's the FAISS index type? | Inner-product (`IndexFlatIP`) on L2-normalized vectors ≡ cosine similarity |
| How many routing features does the ML model use? | 9 (see `FEATURE_NAMES` in `ai_router.py`) |
| What ML algorithm powers the AI router? | `RandomForestClassifier` (scikit-learn) |
| How is cost tracked per request? | Token count × per-model price in `app/services/cost_calculator.py` |
| What does the quality judge do? | Heuristic scoring + 5% of requests get LLM-as-judge evaluation |
| What's the request size limit? | 1MB body, 200 messages max (`RequestBodyLimitMiddleware`) |
| How are plugins invoked? | Lifecycle hooks via `app/core/callback_manager.py` on pre/post request events |
| What port does nginx listen on? | Port 80 internally; published to host as 8000 |
| What database does local dev use? | SQLite (`DATABASE_URL=sqlite:///./optillm.db`) |
| How many CI tests pass? | 166 |
| What's the P50 router overhead? | 1.60 ms |
| What's the cache hit latency? | 8.36 ms |
| What's the TF-IDF compression ratio? | 68.8% token reduction on conversational prompts |

---

*Document generated from direct source code analysis of the OptiLLM repository. All answers are grounded in actual implementation files, not documentation alone.*
