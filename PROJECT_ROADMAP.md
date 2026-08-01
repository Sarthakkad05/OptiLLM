# OptiLLM — Project Roadmap

> **Living Document** — This file is updated at the end of every phase.
> Last updated: 2026-08-01
> Status: **Phase 3 — Complete**




---

## Table of Contents

1. [Vision](#1-vision)
2. [Core Principles](#2-core-principles)
3. [Technology Stack](#3-technology-stack)
4. [Project Features](#4-project-features)
5. [Project Phases](#5-project-phases)
6. [Learning Roadmap](#6-learning-roadmap)
7. [Planned Folder Structure](#7-planned-folder-structure)
8. [Future Vision](#8-future-vision)
9. [Progress Tracker](#9-progress-tracker)
10. [Development Rules](#10-development-rules)

---

## 1. Vision

### What is OptiLLM?

OptiLLM is an **AI Gateway with an Optimization Engine**.

It sits between your application and LLM providers — transparently intercepting every request and applying a stack of optimizations before the request ever reaches a model. After the provider responds, OptiLLM enriches the response with cost data, savings calculations, latency metrics, and quality signals before returning it to the caller.

OptiLLM is **not** a chatbot. It is **not** a model. It is **not** a prompt library.

It is infrastructure — purpose-built to make LLM usage cheaper, faster, more reliable, and more observable at any scale.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Your Application                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │  OpenAI-compatible API
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OptiLLM Gateway                          │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   Auth   │  │  Cache   │  │Compressor│  │    Router    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │Analytics │  │Evaluator │  │ Budgets  │  │   Workflows  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
          ┌─────────┐     ┌─────────┐     ┌─────────┐
          │ OpenAI  │     │ Gemini  │     │Anthropic│
          └─────────┘     └─────────┘     └─────────┘
```

### Why Are We Building It?

Every team using LLMs faces the same problems:

- **Cost is unpredictable.** A single production traffic spike can generate thousands of dollars in LLM bills overnight.
- **Vendor lock-in is real.** Once you hardcode OpenAI's SDK, switching to Gemini or Anthropic requires rewriting infrastructure.
- **Prompts are wasteful.** Most applications send far more tokens than necessary. Conversations accumulate context that no longer contributes to quality.
- **There is no optimization layer.** You get rate limiting, but nothing that actually makes LLM usage smarter.
- **Observability is poor.** Most teams have no answer to "what did our LLMs cost yesterday, and why?".

OptiLLM solves all of these problems at the infrastructure level — so product teams never have to think about them again.

### Who Is the Target Audience?

| Audience | Pain Point OptiLLM Solves |
|---|---|
| **Startups** using LLMs in production | Uncontrolled API spend, no observability |
| **Platform engineers** managing AI infrastructure | No unified gateway across multiple teams and providers |
| **ML engineers** building LLM-powered products | Prompt inefficiency, no evaluation layer |
| **Open-source contributors** | A reference implementation of production AI infrastructure |
| **Enterprises** | Audit logs, RBAC, budget controls, vendor flexibility |

### What Problems Does It Solve?

| Problem | OptiLLM Solution |
|---|---|
| High API cost | Semantic cache, model routing, context compression |
| Vendor lock-in | OpenAI-compatible API with pluggable provider adapters |
| Prompt inefficiency | Two-pass compression: heuristic cleaning + token-aware truncation |
| Unpredictable spend | Per-request cost logging, budget manager, cost prediction |
| No observability | Full request analytics, cost tracking, latency, model distribution |
| Poor routing | Rule-based → AI-driven complexity routing across models |
| No quality signal | LLM-as-judge evaluation engine |
| Cold-start latency | Pre-loaded embedding model + FAISS index at startup |

---

## 2. Core Principles

These principles govern every design decision in this project. When in doubt, return to this list.

### 1. Optimization Before Generation
Every request must pass through the optimization pipeline before reaching a provider. The goal is to reduce the cost and token footprint of every call without degrading response quality.

### 2. OpenAI Compatibility by Default
The gateway must be a drop-in replacement for the OpenAI API. Any client that works with OpenAI must work with OptiLLM with zero code changes beyond changing `base_url`.

### 3. Provider Agnostic
No business logic inside the gateway should know or care about which provider is being called. All provider-specific code lives in isolated adapter modules behind a common interface.

### 4. Modular Architecture
Every feature is a module. Modules have clear interfaces and minimal dependencies. Any module can be disabled, replaced, or extended without modifying others.

### 5. Extensible by Design
New providers, new optimization strategies, new evaluation metrics, and new workflows should be addable without touching the core gateway code.

### 6. Developer Experience First
The project should be runnable locally in under 5 minutes. Documentation must be as important as code. Error messages must be actionable.

### 7. Open Source First
Every architectural decision must be justifiable in a public, collaborative context. Prefer well-known libraries over novel custom implementations where appropriate.

### 8. Performance Oriented
The gateway adds latency to every request. That overhead must be minimal, measurable, and documented. Async-first, connection pooling, and pre-loading are required, not optional.

### 9. Separation of Concerns
The gateway routes. The optimizer optimizes. The evaluator evaluates. The analytics service aggregates. No module should reach into the domain of another.

### 10. Architecture First, Code Second
Every new feature must have a documented design before a line of code is written. Implementation details change; architecture shapes the project.

---

## 3. Technology Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Primary language |
| **FastAPI** | 0.110+ | HTTP framework — async, OpenAPI out of the box |
| **Uvicorn** | 0.27+ | ASGI server |
| **Pydantic** | v2 | Data validation and settings management |
| **pydantic-settings** | 2.2+ | Environment-based configuration |
| **httpx** | 0.27+ | Async HTTP client for provider calls |
| **python-dotenv** | 1.0+ | `.env` loading for local dev |

### AI / ML
| Technology | Purpose |
|---|---|
| **tiktoken** | Token counting for OpenAI-compatible models |
| **sentence-transformers** | Local embedding generation for semantic cache |
| **FAISS** | In-memory vector similarity search |
| **LangChain** *(Phase 7)* | Prompt templates, document loaders, structured output |
| **LangGraph** *(Phase 8)* | Stateful multi-step workflow orchestration |

### Database
| Technology | Purpose |
|---|---|
| **SQLite** | Default local development database (zero-config) |
| **PostgreSQL** | Production database |
| **SQLAlchemy** | ORM and query builder |
| **Alembic** *(Phase 13)* | Database schema migrations for production |

### Cache
| Technology | Purpose |
|---|---|
| **FAISS** | Vector similarity index (in-process, persisted to disk) |
| **Redis** *(Phase 5+)* | Distributed semantic cache, rate limiting, session store |

### Vector Database
| Technology | Purpose |
|---|---|
| **FAISS** | MVP — fast, local, no external dependency |
| **Qdrant** *(Phase 12+)* | Production-grade vector DB for RAG and scaled cache |

### Frontend *(Phase 13)*
| Technology | Purpose |
|---|---|
| **Next.js** | Dashboard UI |
| **Recharts / Tremor** | Analytics charts |
| **TailwindCSS** | Styling |

### Infrastructure
| Technology | Purpose |
|---|---|
| **Docker** | Containerization |
| **Docker Compose** | Local multi-service orchestration |
| **Kubernetes** *(Phase 13)* | Production orchestration |
| **AWS / GCP** *(Phase 13)* | Cloud deployment targets |

### Observability
| Technology | Purpose |
|---|---|
| **OpenTelemetry** *(Phase 13)* | Distributed tracing standard |
| **Prometheus** *(Phase 13)* | Metrics collection |
| **Grafana** *(Phase 13)* | Metrics dashboards |
| **Python logging** | Structured application logs (current) |

### Testing
| Technology | Purpose |
|---|---|
| **pytest** | Unit and integration tests |
| **httpx (AsyncClient)** | Async endpoint testing |
| **pytest-asyncio** | Async test support |
| **factory_boy** | Test fixture factories |

---

## 4. Project Features

### 4.1 Gateway

| Feature | Description | Status |
|---|---|---|
| OpenAI-compatible endpoint | `POST /v1/chat/completions` drop-in replacement | ✅ Done |
| Health check | `GET /health` with DB connection status | ✅ Done |
| CORS middleware | Configurable allowed origins | ✅ Done |
| Async request handling | Full async stack via FastAPI + httpx | ✅ Done |
| Provider fallback | Auto-switch to backup provider on failure | ✅ Done |
| Retry with backoff | Up to 3 retry attempts with exponential backoff | ✅ Done |
| Mock mode | Simulated responses when no API keys configured | ✅ Done |
| Request ID tracing | Unique ID on every request for log correlation | ✅ Done |
| Multi-provider dispatch | Route to OpenAI or Gemini based on model name | ✅ Done |
| Streaming support | Server-sent events for streaming responses | ✅ Done |
| API key authentication | Bearer token validation middleware | ✅ Done |
| Rate limiting | Per-key and per-IP request limits | ✅ Done |


### 4.2 Optimization Engine

| Feature | Description | Status |
|---|---|---|
| Semantic cache (FAISS) | Local vector similarity cache with TTL | ✅ Done |
| Cache sync on restart | Rebuild FAISS from DB to fix ID-drift | ✅ Done |
| Heuristic compression | Strip whitespace, dedup lines, remove redundant markdown | ✅ Done |
| Token-aware truncation | Center-truncation preserving start + end | ✅ Done |
| Token counting | tiktoken-based pre-request token count | ✅ Done |
| Cost estimation | Per-model USD cost from pricing table | ✅ Done |
| Compression savings calc | Estimate savings from token reduction | ✅ Done |
| Routing savings calc | Estimate savings from model downgrade | ✅ Done |
| Cache savings calc | Estimate savings from cache hit | ✅ Done |
| Bypass flags | Per-request `bypass_cache/compression/routing` | ✅ Done |
| Distributed cache (Redis) | Redis-backed semantic cache for horizontal scale | 🔲 Phase 5 |
| Budget manager | Per-key/per-day spending caps | 🔲 Phase 6 |
| Cost prediction | Predict request cost before calling provider | 🔲 Phase 10 |
| Prompt summarization | LLM-powered conversation summarization | 🔲 Phase 6 |

### 4.3 Model Routing

| Feature | Description | Status |
|---|---|---|
| Rule-based router | Complexity scoring across 4 dimensions | ✅ Done |
| Routing table | LOW/MEDIUM/HIGH → model mapping | ✅ Done |
| Runtime config update | Change routing table without restart | ✅ Done |
| Safety fallback | Gemini → OpenAI if Gemini key missing | ✅ Done |
| Load balancing | Round-robin across provider replicas | 🔲 Phase 3 |
| Provider health checks | Remove failing providers from rotation | 🔲 Phase 3 |
| AI-driven routing | LLM classifier to predict best model | 🔲 Phase 10 |
| Historical routing | Choose model based on past performance | 🔲 Phase 10 |
| Benchmark-driven routing | Route by quality benchmark score | 🔲 Phase 10 |
| Adaptive routing | Self-tuning based on real outcomes | 🔲 Phase 10 |

### 4.4 Analytics

| Feature | Description | Status |
|---|---|---|
| Request log (DB) | Every request persisted to `request_logs` | ✅ Done |
| KPI summary | Total requests, cache hit rate, cost, savings | ✅ Done |
| Cost over time | Daily aggregated cost series | ✅ Done |
| Model distribution | Per-model request counts | ✅ Done |
| Recent requests | Last 50 requests with full metadata | ✅ Done |
| Compression stats | Aggregated token savings and compression rate | ✅ Done |
| Routing stats | Routing rate, model breakdown, savings | ✅ Done |
| Cache stats | FAISS vector count, active/expired entries | ✅ Done |
| Cache clear endpoint | Wipe cache and reset FAISS | ✅ Done |
| Prompt snippet logging | First 200 chars stored for debugging | ✅ Done |
| Token usage charts | Input/output/saved token trends | 🔲 Phase 4 |
| Latency percentiles | p50/p95/p99 latency per provider | 🔲 Phase 4 |
| Provider comparison | Side-by-side cost and latency by provider | 🔲 Phase 4 |
| OpenTelemetry tracing | Distributed trace per request | 🔲 Phase 13 |

### 4.5 Intelligence (Future Phases)

| Feature | Description | Phase |
|---|---|---|
| Prompt classification | Classify prompt by task type | Phase 9 |
| LangGraph workflows | Multi-step optimization pipelines | Phase 8 |
| Tool calling | Structured function calling in routing decisions | Phase 9 |
| Conditional routing | Graph-based routing with conditional edges | Phase 8 |
| Context summarization | Summarize long conversation history with LLM | Phase 6 |

### 4.6 Evaluation (Future Phases)

| Feature | Description | Phase |
|---|---|---|
| LLM-as-judge | Use a small model to score response quality | Phase 11 |
| Cost vs quality scoring | Composite score: quality / cost ratio | Phase 11 |
| Hallucination detection | Detect factual inconsistencies in responses | Phase 11 |
| Benchmark runner | Run MMLU, HumanEval, etc. per-model | Phase 11 |
| Response comparison | A/B test two models on the same prompt | Phase 11 |

### 4.7 Future Features

| Feature | Phase |
|---|---|
| RAG support (chunking, retrieval, hybrid search) | Phase 12 |
| Qdrant integration | Phase 12 |
| Multi-tenant RBAC | Phase 14 |
| API key management UI | Phase 14 |
| Audit logs | Phase 14 |
| Encryption at rest | Phase 14 |
| CLI (`optillm run`, `optillm status`) | Phase 13 |
| Python SDK (`pip install optillm-client`) | Phase 13 |
| Plugin system | Phase 14 |
| MCP server integration | Future |
| Local model support (Ollama) | Future |
| Enterprise SLA manager | Future |

---

## 5. Project Phases

### Phase 0 — Cleanup & Architecture Refactor

**Status: ✅ Complete**

**Goal:** Transform the experimental prototype into a clean, maintainable codebase with a solid foundation.

**Deliverables:**
- Remove all demo, dashboard, and prototype code
- Remove unused dependencies (Streamlit, Pandas, Alembic)
- Migrate `@on_event` lifecycle to `lifespan` context manager
- Eliminate hardcoded constants (FAISS path, database URL)
- Fix Pydantic v2 namespace warnings
- Fix enum serialization bug (`Complexity.LOW` → `"low"`)
- Create comprehensive README and PROJECT_ROADMAP
- Expand `.gitignore`

**Technologies Introduced:** None (cleanup only)

**Completion Criteria:**
- `python -c "from app.main import app"` produces zero warnings
- `GET /health` returns `{"status": "ok"}`
- `POST /v1/chat/completions` returns valid response with `optillm_metadata`
- No prototype files remain in the repository

---

### Phase 1 — Gateway Foundation

**Status: ✅ Complete**


**Goal:** Harden the gateway core with streaming, authentication, testing infrastructure, and structured error handling.

**Deliverables:**
- SSE streaming for `POST /v1/chat/completions`
- API key authentication middleware (Bearer token)
- Structured error responses (RFC 7807 Problem Details)
- Request/response logging middleware
- Test suite: `pytest` + `httpx.AsyncClient` for all endpoints
- Unit tests for `compressor.py`, `router.py`, `cost_estimator.py`, `token_counter.py`
- Code quality tooling (`ruff`, `black`, `isort`)
- In-memory rate limiter middleware

**Technologies Introduced:**
- `pytest`, `pytest-asyncio`, `httpx.AsyncClient`
- `ruff` (linting), `black` (formatting), `isort`


**Completion Criteria:**
- Streaming responses work end-to-end with a real OpenAI key
- Every endpoint has at least one integration test
- All tests pass in CI with zero flaky failures
- Gateway returns RFC 7807 structured errors, not raw 500s

---

### Phase 2 — Multi-Provider Support

**Status: ✅ Complete**


**Goal:** Expand the provider layer to support Anthropic and formalize the adapter pattern so adding new providers is trivial.

**Deliverables:**
- `BaseProvider` abstract interface (common contract for all providers)
- Refactor `openai_client.py` and `gemini_client.py` to implement `BaseProvider`
- `AnthropicProvider` implementation (Claude 3 family)
- Provider registry: `{"openai": OpenAIProvider, "gemini": GeminiProvider, ...}`
- Replace `if/elif` dispatch with registry lookup in `dispatcher.py`
- `ALLOWED_PROVIDERS`, `ALLOWED_MODELS`, `ALLOWED_ORIGINS` config
- Provider-level timeout configuration

**New Config Variables:**
```
ANTHROPIC_API_KEY
ALLOWED_PROVIDERS
ALLOWED_MODELS
ALLOWED_ORIGINS
REQUEST_TIMEOUT_SECONDS
```

**Completion Criteria:**
- Adding a new provider requires creating one file and registering it — zero gateway changes
- Claude models work identically to OpenAI models from the client perspective
- All three providers have integration tests

---

### Phase 3 — Routing & High Availability

**Status: ✅ Complete**


**Goal:** Make the gateway resilient under provider failure. Add load balancing, circuit breaking, and real-time health monitoring.

**Deliverables:**
- Provider health check background task
- Circuit breaker: auto-remove unhealthy providers from rotation
- Round-robin load balancing across multiple provider API keys
- `GET /api/v1/providers/status` real-time health endpoint
- `ROUTING_STRATEGY` config: `round_robin | least_latency | cost_optimized`
- Weighted provider routing

**Completion Criteria:**
- Provider failure causes zero gateway downtime
- `/providers/status` reflects real-time health accurately
- Load balancing distributes traffic within 5% of configured weights

---

### Phase 4 — Analytics Engine

**Status: 🔲 Not Started**

**Goal:** Build a rich analytics layer that answers "what did our LLMs cost, where, when, and why?"

**Deliverables:**
- Latency percentiles: `GET /api/v1/analytics/latency` (p50/p95/p99)
- Token trends: `GET /api/v1/analytics/tokens`
- Provider analytics: `GET /api/v1/analytics/providers`
- Savings breakdown: `GET /api/v1/analytics/savings`
- Request tagging via `x-optillm-tag` header
- Date range filtering on all analytics endpoints
- DB indexes on all analytics-heavy columns

**Completion Criteria:**
- All analytics queries complete in < 200ms at 100k row scale
- Every savings dimension independently queryable

---

### Phase 5 — Distributed Semantic Cache

**Status: 🔲 Not Started**

**Goal:** Replace the in-process FAISS cache with a distributed, horizontally scalable cache backed by Redis.

**Architecture:**
```
Request → Embedding → Redis similarity lookup
                          ↓ miss
                      FAISS fallback (single instance)
                          ↓ miss
                      LLM Provider → Insert into Redis + FAISS
```

**Deliverables:**
- Redis integration (`redis-py`, `REDIS_URL` config)
- Redis-backed semantic cache with TTL
- Cache namespace support (per-tenant isolation)
- Graceful degradation if Redis is unreachable
- Cache warming endpoint
- FAISS retained as single-instance fallback

**New Config Variables:**
```
REDIS_URL
CACHE_TTL_SECONDS
CACHE_NAMESPACE
CACHE_SIMILARITY_THRESHOLD
```

**Completion Criteria:**
- Cache works across multiple gateway instances
- Redis failure produces zero errors in the request pipeline
- Cache hit rate is measured and reported in analytics

---

### Phase 6 — Prompt Optimization Engine

**Status: 🔲 Not Started**

**Goal:** Build conversation summarization, duplicate detection, and a budget management system.

**Optimization Strategies:**

| Strategy | When Applied | Expected Savings |
|---|---|---|
| Heuristic cleaning | Always | 2–8% tokens |
| Center truncation | > 2,000 tokens | 15–40% tokens |
| Conversation summarization | > 8 turns | 30–60% tokens |
| Duplicate removal | Repeated messages | Varies |

**Deliverables:**
- Conversation summarization (cheap model replaces long history)
- Duplicate message detection and removal
- Budget manager: per-key daily/monthly spend caps
- Cost prediction pre-flight check
- Budget enforcement middleware
- `GET /api/v1/budgets` and `POST /api/v1/budgets` endpoints

**Completion Criteria:**
- Budget manager blocks requests at configured limits accurately
- Summarization preserves factual accuracy (verified on test set)
- Cost prediction is within 10% of actual cost

---

### Phase 7 — LangChain Integration

**Status: 🔲 Not Started**

**Goal:** Integrate LangChain as a utility library for prompt templates and structured output.

> **Important:** LangChain is a **tool**, not the architecture. It is only used where its specific capabilities (e.g., `ChatPromptTemplate`, `PydanticOutputParser`) clearly outperform a custom implementation.

**Where LangChain IS used:**
- Prompt template management and variable injection
- Structured output parsing
- Evaluation chains (Phase 11)
- Document loaders (Phase 12)

**Where LangChain is NOT used:**
- The main request pipeline
- Provider HTTP calls (we keep direct httpx)
- Routing decisions
- Caching

**Deliverables:**
- `PromptTemplateEngine` with template versioning
- `StructuredOutputParser` backed by Pydantic
- Prompt template registry (store/retrieve by name + version)
- `POST /api/v1/prompts/render` endpoint

---

### Phase 8 — LangGraph Workflow Engine

**Status: 🔲 Not Started**

**Goal:** Replace the linear `process_request()` pipeline with a graph-based workflow that supports conditional paths and parallel branches.

**Why LangGraph?**

The current pipeline is a fixed linear sequence. LangGraph enables:
- Conditional retry if quality score is too low
- Parallel model comparison (run two models, return the better one)
- Multi-step optimization depending on intermediate results

**Planned Workflow Graph:**
```
Receive Request
      │
      ▼
  Cache Check ──HIT──► Return Response
      │ MISS
      ▼
  Compress Context
      │
      ▼
  Classify Complexity
      │
      ▼
  Route to Model
      │
      ▼
  Call Provider
      │
      ▼
  Quality Check ──FAIL (retries left)──► Route to Model
      │ PASS
      ▼
  Insert Cache
      │
      ▼
  Log Analytics
      │
      ▼
  Return Response
```

**Deliverables:**
- `GatewayState` typed state object
- `OptimizationGraph` LangGraph state graph
- Quality retry conditional edge
- Parallel dual-model comparison branch
- `POST /api/v1/workflows` endpoint
- Workflow state inspection API

**Completion Criteria:**
- Quality retry loop works end-to-end
- Graph overhead vs linear pipeline < 5ms
- Graph state is inspectable via API

---

### Phase 9 — Tool Calling

**Status: 🔲 Not Started**

**Goal:** Give routing and optimization decisions access to real-time information via a structured tool calling system.

**Tool Registry (Planned):**

| Tool | Returns | Used By |
|---|---|---|
| `get_provider_latency(provider)` | float (ms) | Router |
| `get_provider_cost(model, tokens)` | float (USD) | Optimizer |
| `get_cache_hit_rate(namespace)` | float (0–1) | Cache strategy |
| `get_budget_remaining(api_key)` | float (USD) | Budget enforcer |
| `classify_prompt(text)` | TaskType | AI Router |

**Security Constraints:**
- Tools cannot make outbound HTTP calls unless explicitly allowlisted
- Execution timeout: 2 seconds maximum
- Read-only by default

**Deliverables:**
- Tool registry with decorator-based registration
- Tool execution sandbox
- `GET /api/v1/tools` endpoint
- Tool call audit log

---

### Phase 10 — Intelligent AI Routing

**Status: 🔲 Not Started**

**Goal:** Replace the static rule-based complexity scorer with an ML classifier that learns from historical outcomes.

**Routing Evolution:**

| Version | Method | Accuracy | Overhead |
|---|---|---|---|
| v1 (current) | Rule-based score (4 signals) | ~70% | < 1ms |
| v2 (Phase 10) | ML classifier (trained on logs) | ~85% | ~5ms |
| v3 (future) | Adaptive router (online learning) | ~92% | ~10ms |

**Deliverables:**
- Historical routing data export from analytics
- Feature engineering pipeline
- Lightweight classifier (logistic regression or small transformer)
- Shadow mode: run AI router in parallel, log disagreements
- Confidence threshold fallback to rule-based router
- `GET /api/v1/router/explain` explainability endpoint
- Benchmark-driven routing table

**Completion Criteria:**
- AI router achieves > 80% agreement with expert routing decisions
- Routing overhead < 10ms
- Shadow mode identifies rule-based failures correctly

---

### Phase 11 — Evaluation Engine

**Status: 🔲 Not Started**

**Goal:** Score response quality on every request and feed those scores back into routing and optimization decisions.

**Evaluation Dimensions:**

| Dimension | Description |
|---|---|
| Correctness | Is the answer factually accurate? |
| Relevance | Does the answer address the question? |
| Completeness | Is the answer sufficiently detailed? |
| Conciseness | Is there unnecessary verbosity? |
| Safety | Does the answer contain harmful content? |

**Deliverables:**
- `LLMJudge` using a cheap model (e.g., `gemini-2.0-flash`) as evaluator
- `ResponseComparison` A/B testing endpoint
- `HallucinationDetector`
- Quality score stored on `RequestLog`
- `GET /api/v1/analytics/quality` endpoint
- Cost-vs-quality efficiency score: `quality / cost_usd`

**Completion Criteria:**
- Quality scores correlate with human ratings on 100 annotated requests
- A/B comparison endpoint usable from the dashboard

---

### Phase 12 — RAG Support

**Status: 🔲 Not Started**

**Goal:** First-class Retrieval-Augmented Generation — document ingestion, chunking, embedding, and retrieval behind the gateway API.

**Deliverables:**
- `DocumentPipeline`: ingest PDF, DOCX, TXT, Markdown
- Chunking strategies: fixed-size, semantic, recursive character
- Qdrant integration as production vector store
- Hybrid search: BM25 keyword + dense vector retrieval
- `POST /api/v1/rag/ingest` — upload and index a document
- `POST /api/v1/rag/query` — retrieve relevant chunks
- RAG-augmented completions: auto-inject retrieved context
- `GET /api/v1/rag/collections` — list document collections

**Completion Criteria:**
- End-to-end RAG pipeline works on a 100-page PDF
- Retrieval latency < 100ms at 10,000 chunks
- RAG responses measurably more accurate on domain-specific queries

---

### Phase 13 — Production Infrastructure

**Status: 🔲 Not Started**

**Goal:** Make OptiLLM production-grade with Kubernetes deployment, observability, a dashboard, and a CLI.

**Deliverables:**
- Alembic migrations (replace `create_all()`)
- Kubernetes manifests (Deployment, Service, ConfigMap, Secrets)
- Helm chart for one-command deployment
- OpenTelemetry distributed tracing
- Prometheus metrics exporter (`/metrics`)
- Grafana dashboard configuration
- `optillm` CLI: `start`, `status`, `logs`, `metrics`, `cache clear`
- Next.js analytics dashboard
- Horizontal pod autoscaling
- Python SDK package (`pip install optillm-client`)

**Completion Criteria:**
- Single `helm install` deploys the full stack
- Dashboard shows real-time analytics
- P99 gateway overhead < 50ms at 100 RPS

---

### Phase 14 — Enterprise Features

**Status: 🔲 Not Started**

**Goal:** Multi-tenancy, RBAC, audit logs, encryption, and a plugin system.

**Deliverables:**
- Multi-tenant namespace isolation
- RBAC: admin, developer, read-only roles
- API key management UI
- Immutable audit log
- Encryption at rest for prompt/response data
- PII detection and redaction middleware
- Plugin system: `BasePlugin` interface + registry + dynamic loader
- SLA manager: per-tenant latency commitments
- Reference plugin implementation

**Completion Criteria:**
- Tenants cannot access each other's data (verified by integration tests)
- Audit log is tamper-evident
- Plugin API is documented with one working reference plugin

---

## 6. Learning Roadmap

Each phase introduces and deepens specific GenAI and infrastructure knowledge.

| Phase | Core Topics Learned |
|---|---|
| **Phase 0** | FastAPI lifecycle, Pydantic v2, FAISS basics, clean architecture |
| **Phase 1** | SSE streaming, async middleware, integration testing, CI/CD |
| **Phase 2** | LLM provider APIs, adapter pattern, API normalization across providers |
| **Phase 3** | Circuit breaking, health monitoring, load balancing strategies |
| **Phase 4** | Time-series analytics, SQL aggregation, query optimization |
| **Phase 5** | Embeddings deep dive, vector similarity, Redis data structures, cache invalidation |
| **Phase 6** | Token economics, prompt engineering, conversation summarization, budget systems |
| **Phase 7** | LangChain prompt templates, structured output parsing, evaluation chains |
| **Phase 8** | LangGraph state machines, conditional DAGs, graph-based orchestration |
| **Phase 9** | Tool calling protocols, function schemas, sandboxed execution |
| **Phase 10** | ML feature engineering, classification, online learning, explainability |
| **Phase 11** | LLM-as-judge, hallucination detection, quality metrics, A/B testing |
| **Phase 12** | RAG architecture, chunking strategies, hybrid search, Qdrant |
| **Phase 13** | Kubernetes, Helm, OpenTelemetry, Prometheus, Grafana, Next.js |
| **Phase 14** | Multi-tenancy, RBAC, encryption, plugin systems, compliance |

### Cumulative GenAI Knowledge Map

```
Foundational               Intermediate                Advanced
──────────────────         ──────────────────────      ──────────────────────────
• LLM API calls            • Semantic similarity        • AI-driven routing
• Token counting           • Vector search (FAISS)      • Adaptive routing
• Cost estimation          • RAG pipelines              • LLM-as-judge
• Prompt structure         • Prompt compression         • Fine-tuned classifiers
• Provider differences     • LangChain abstractions     • Plugin architectures
• Embeddings basics        • LangGraph workflows        • Multi-tenant LLM infra
                           • Tool calling               • Evaluation at scale
                           • Evaluation metrics         • Hallucination detection
```

---

## 7. Planned Folder Structure

The following is the full target structure for a completed project. Folders are added progressively as phases are completed. Do not create empty folders in advance.

```
OptiLLM/
│
├── app/                            # FastAPI application root
│   ├── main.py                     # App factory + lifespan context
│   │
│   ├── api/                        # HTTP layer — thin routing only
│   │   ├── api.py                  # Aggregates all routers
│   │   ├── dependencies.py         # Shared FastAPI dependencies
│   │   └── endpoints/
│   │       ├── health.py           # GET /health
│   │       ├── proxy.py            # POST /v1/chat/completions
│   │       ├── analytics.py        # GET /api/v1/analytics/*
│   │       ├── cache.py            # GET/DELETE /api/v1/cache/*
│   │       ├── router_config.py    # GET/POST /api/v1/router/*
│   │       ├── providers.py        # GET /api/v1/providers/* (Phase 3)
│   │       ├── prompts.py          # POST /api/v1/prompts/* (Phase 7)
│   │       ├── workflows.py        # POST /api/v1/workflows/* (Phase 8)
│   │       ├── tools.py            # GET/POST /api/v1/tools/* (Phase 9)
│   │       ├── budgets.py          # GET/POST /api/v1/budgets/* (Phase 6)
│   │       ├── rag.py              # POST /api/v1/rag/* (Phase 12)
│   │       └── evaluation.py       # POST /api/v1/evaluate/* (Phase 11)
│   │
│   ├── core/                       # Cross-cutting concerns
│   │   ├── config.py               # All env vars — single source of truth
│   │   ├── logging.py              # Structured logging setup
│   │   ├── exceptions.py           # Custom exception hierarchy (Phase 1)
│   │   └── constants.py            # Project-wide constants (Phase 1)
│   │
│   ├── middleware/                 # ASGI middleware (Phase 1+)
│   │   ├── auth.py                 # API key authentication
│   │   ├── rate_limit.py           # Rate limiting
│   │   ├── request_id.py           # X-Request-ID injection
│   │   ├── pii_redaction.py        # PII detection (Phase 14)
│   │   └── audit_log.py            # Audit log capture (Phase 14)
│   │
│   ├── db/                         # Database layer
│   │   ├── base_class.py           # SQLAlchemy declarative base
│   │   ├── base.py                 # Model import aggregator
│   │   ├── session.py              # Engine + session factory
│   │   └── models.py               # All SQLAlchemy models
│   │
│   ├── engine/                     # Optimization engine — core IP
│   │   ├── cache.py                # Semantic cache orchestration
│   │   ├── compressor.py           # Context compression pipeline
│   │   ├── embedding.py            # SentenceTransformer singleton
│   │   ├── faiss_store.py          # Local FAISS index management
│   │   ├── redis_cache.py          # Redis-backed cache (Phase 5)
│   │   ├── router.py               # Rule-based model routing
│   │   ├── ai_router.py            # ML-based routing (Phase 10)
│   │   └── budget.py               # Spend tracking (Phase 6)
│   │
│   ├── providers/                  # LLM provider adapters
│   │   ├── base.py                 # BaseProvider abstract class
│   │   ├── registry.py             # Provider registration + lookup
│   │   ├── dispatcher.py           # Retry + fallback orchestrator
│   │   ├── openai_client.py        # OpenAI adapter
│   │   ├── gemini_client.py        # Google Gemini adapter
│   │   ├── anthropic_client.py     # Anthropic adapter (Phase 2)
│   │   └── ollama_client.py        # Local Ollama adapter (Future)
│   │
│   ├── workflows/                  # LangGraph workflow graphs (Phase 8)
│   │   ├── optimization_graph.py   # Main optimization DAG
│   │   ├── rag_graph.py            # RAG retrieval + generation
│   │   └── evaluation_graph.py     # Quality evaluation workflow
│   │
│   ├── langchain/                  # LangChain utilities (Phase 7)
│   │   ├── prompt_engine.py        # Prompt template management
│   │   ├── output_parser.py        # Structured output parsing
│   │   └── eval_chains.py          # Evaluation chains
│   │
│   ├── evaluation/                 # Evaluation engine (Phase 11)
│   │   ├── judge.py                # LLM-as-judge
│   │   ├── hallucination.py        # Hallucination detection
│   │   ├── benchmarks.py           # Benchmark runner
│   │   └── scorer.py               # Quality / cost efficiency score
│   │
│   ├── rag/                        # RAG pipeline (Phase 12)
│   │   ├── ingestion.py            # Document loading + chunking
│   │   ├── embedder.py             # Embedding pipeline
│   │   ├── retrieval.py            # Hybrid BM25 + dense search
│   │   └── qdrant_store.py         # Qdrant adapter
│   │
│   ├── schemas/                    # Pydantic request/response schemas
│   │   ├── chat.py                 # Chat completion schemas
│   │   ├── analytics.py            # Analytics schemas
│   │   ├── provider.py             # Provider health schemas (Phase 3)
│   │   ├── evaluation.py           # Evaluation schemas (Phase 11)
│   │   ├── rag.py                  # RAG schemas (Phase 12)
│   │   └── workflow.py             # Workflow schemas (Phase 8)
│   │
│   ├── services/                   # Business logic
│   │   ├── gateway.py              # Main request pipeline orchestrator
│   │   ├── analytics.py            # Analytics aggregation
│   │   ├── cost_estimator.py       # USD cost calculation
│   │   ├── token_counter.py        # tiktoken-based token counting
│   │   ├── provider_health.py      # Health check background task (Phase 3)
│   │   └── summarizer.py           # Conversation summarization (Phase 6)
│   │
│   └── plugins/                    # Plugin system (Phase 14)
│       ├── base_plugin.py          # BasePlugin interface
│       ├── registry.py             # Plugin registry
│       └── loader.py               # Dynamic plugin loading
│
├── tests/                          # Test suite
│   ├── conftest.py                 # Shared fixtures
│   ├── unit/                       # Isolated function tests (no I/O)
│   │   ├── test_compressor.py
│   │   ├── test_router.py
│   │   ├── test_cost_estimator.py
│   │   └── test_token_counter.py
│   ├── integration/                # Full HTTP cycle tests
│   │   ├── test_proxy_endpoint.py
│   │   ├── test_analytics_endpoint.py
│   │   └── test_health_endpoint.py
│   └── e2e/                        # End-to-end tests (Phase 13)
│       └── test_full_pipeline.py
│
├── migrations/                     # Alembic migrations (Phase 13)
│
├── dashboard/                      # Next.js frontend (Phase 13)
│
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── api.md
│   ├── providers.md
│   ├── configuration.md
│   └── contributing.md
│
├── scripts/                        # Utility scripts
│   ├── seed_cache.py
│   ├── export_analytics.py
│   └── benchmark_providers.py
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   └── release.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── PROJECT_ROADMAP.md
└── README.md
```

### Folder Rationale

| Folder | Why It Exists |
|---|---|
| `app/api/` | HTTP routing only. Endpoints call one service, return one response. Zero business logic. |
| `app/engine/` | The core intellectual property: cache, compressor, router. Heavily tested. |
| `app/providers/` | Provider-specific code is completely isolated here. Nothing outside knows which HTTP API is called. |
| `app/services/` | Coordinates between engine, providers, and database. Contains the pipeline orchestration. |
| `app/middleware/` | Cross-cutting concerns without polluting endpoint functions. |
| `app/workflows/` | LangGraph graphs have their own state management — isolated from services. |
| `app/langchain/` | LangChain is a utility dependency, not the architecture. Isolated for easy replacement. |
| `tests/unit/` | Test individual functions with no I/O, no DB, no network calls. |
| `tests/integration/` | Test full HTTP request/response cycles with a real (test) database. |
| `docs/` | Documentation is a first-class deliverable, co-located with code. |

---

## 8. Future Vision

### Version 1.0 — The Production Gateway

By v1.0, OptiLLM is a complete production AI Gateway:

- Full optimization pipeline (cache, compression, routing)
- Multi-provider: OpenAI, Gemini, Anthropic, Ollama
- LangGraph workflow engine
- LLM-as-judge evaluation engine
- Redis-backed distributed cache
- REST API, Python SDK, and CLI
- Real-time analytics dashboard (Next.js)
- Kubernetes deployment with Helm
- 80%+ test coverage

### Version 2.0 — The Intelligence Platform

By v2.0, OptiLLM moves from optimization to intelligence:

- AI-driven routing that learns from historical data
- Self-improving model selection based on real quality/cost outcomes
- RAG support with Qdrant
- Plugin marketplace: community-contributed optimizers and evaluators
- Multi-tenant RBAC for enterprise teams
- Fine-tuning integration: route to custom fine-tuned models

### Version 3.0 — The Agent Infrastructure

By v3.0, OptiLLM becomes a complete AI infrastructure platform:

- Agent workflow support: long-running, multi-step autonomous tasks
- MCP (Model Context Protocol) server compatibility
- Benchmark-driven automatic model selection
- Federated deployment: cloud + on-prem hybrid
- Enterprise SLA commitments with automatic failover
- Autonomous cost optimization: gateway rewrites prompts to save cost

### Five-Year Vision

OptiLLM becomes the infrastructure layer for AI systems the way Nginx became the infrastructure layer for web applications — invisible, essential, and everywhere.

Every LLM API call in a well-run engineering organization passes through a gateway like OptiLLM. Our advantage: open-source, optimization-first, built by practitioners who use it themselves.

---

## 9. Progress Tracker

> Update this section at the end of every phase.

### ✅ Phase 0 — Cleanup

- [x] Delete `dashboard/` directory
- [x] Delete `demos/` directory
- [x] Delete `usecase/` directory
- [x] Delete `optillm_client/` directory
- [x] Delete `alembic/` and `alembic.ini`
- [x] Remove runtime artefacts (`optillm.db`, `faiss_store/index.faiss`)
- [x] Remove planning docs (`implementation_plan.md`, `interview.md`, `phases.md`)
- [x] Remove `streamlit`, `pandas`, `alembic` from `requirements.txt`
- [x] Migrate `@on_event` to `lifespan` context manager
- [x] Add `FAISS_INDEX_PATH` to settings
- [x] Add SQLite default for `DATABASE_URL`
- [x] Fix Pydantic v2 namespace warnings (zero startup warnings)
- [x] Fix `Complexity.LOW` enum serialization (`"low"` in API response)
- [x] Create `README.md`
- [x] Expand `.gitignore`
- [x] Create `PROJECT_ROADMAP.md`

---

### ✅ Phase 1 — Gateway Foundation

- [x] Implement SSE streaming (`stream=true` on `/v1/chat/completions`)
- [x] Add API key authentication middleware
- [x] Implement RFC 7807 structured error responses
- [x] Add request/response logging middleware
- [x] Write unit tests for `compressor.py`
- [x] Write unit tests for `router.py`
- [x] Write unit tests for `cost_estimator.py`
- [x] Write unit tests for `token_counter.py`
- [x] Write integration tests for `POST /v1/chat/completions`
- [x] Write integration tests for `GET /health`
- [x] Write integration tests for `GET /api/v1/analytics`
- [x] Add `ruff` and `black` code quality tools
- [x] Add in-memory rate limiter middleware



---

### ✅ Phase 2 — Multi-Provider

- [x] Define `BaseProvider` abstract interface
- [x] Refactor `openai_client.py` to implement `BaseProvider`
- [x] Refactor `gemini_client.py` to implement `BaseProvider`
- [x] Implement `AnthropicProvider`
- [x] Build provider registry
- [x] Refactor `dispatcher.py` to use registry
- [x] Add `ALLOWED_PROVIDERS` config
- [x] Add `ALLOWED_MODELS` config
- [x] Add `ALLOWED_ORIGINS` config
- [x] Integration tests for Anthropic provider


---

### ✅ Phase 3 — Routing & HA

- [x] Provider health check background task
- [x] Circuit breaker implementation
- [x] Round-robin load balancing
- [x] `GET /api/v1/providers/status` endpoint
- [x] `ROUTING_STRATEGY` config variable
- [x] Least-latency & cost-optimized routing
- [x] Failover & status integration tests


---

### 🔲 Phase 4 — Analytics Engine

- [ ] `GET /api/v1/analytics/latency` (p50/p95/p99)
- [ ] `GET /api/v1/analytics/tokens`
- [ ] `GET /api/v1/analytics/providers`
- [ ] `GET /api/v1/analytics/savings`
- [ ] Request tagging via `x-optillm-tag`
- [ ] Date range filtering
- [ ] DB indexes on analytics columns
- [ ] Query performance verified < 200ms at 100k rows

---

### 🔲 Phase 5 — Distributed Cache

- [ ] Redis integration
- [ ] Redis-backed semantic cache
- [ ] TTL configuration
- [ ] Namespace support
- [ ] Graceful Redis degradation
- [ ] Cache warming endpoint
- [ ] Multi-instance cache test

---

### 🔲 Phase 6 — Prompt Optimization

- [ ] Conversation summarization
- [ ] Duplicate message detection
- [ ] Budget manager
- [ ] Cost prediction pre-flight
- [ ] Budget enforcement middleware
- [ ] `GET /api/v1/budgets`
- [ ] `POST /api/v1/budgets`

---

### 🔲 Phase 7 — LangChain

- [ ] `PromptTemplateEngine` with versioning
- [ ] `StructuredOutputParser`
- [ ] Prompt template registry
- [ ] `POST /api/v1/prompts/render`

---

### 🔲 Phase 8 — LangGraph

- [ ] `GatewayState` typed state
- [ ] `OptimizationGraph` DAG
- [ ] Quality retry node
- [ ] Parallel model comparison branch
- [ ] `POST /api/v1/workflows`
- [ ] State inspection API

---

### 🔲 Phase 9 — Tool Calling

- [ ] Tool registry
- [ ] Tool execution sandbox
- [ ] Core gateway tools implemented
- [ ] `GET /api/v1/tools`
- [ ] Tool call audit log

---

### 🔲 Phase 10 — AI Routing

- [ ] Historical routing data export
- [ ] Feature engineering pipeline
- [ ] Lightweight classifier trained
- [ ] Shadow mode (parallel AI + rule-based)
- [ ] Confidence threshold fallback
- [ ] `GET /api/v1/router/explain`
- [ ] Benchmark-driven routing

---

### 🔲 Phase 11 — Evaluation

- [ ] `LLMJudge` (multi-dimension quality scoring)
- [ ] `ResponseComparison` A/B endpoint
- [ ] `HallucinationDetector`
- [ ] Quality score on `RequestLog`
- [ ] `GET /api/v1/analytics/quality`
- [ ] Efficiency score (quality / cost)

---

### 🔲 Phase 12 — RAG

- [ ] `DocumentPipeline` (PDF, DOCX, TXT, Markdown)
- [ ] Chunking strategies
- [ ] Qdrant integration
- [ ] Hybrid search
- [ ] `POST /api/v1/rag/ingest`
- [ ] `POST /api/v1/rag/query`
- [ ] RAG-augmented completions
- [ ] `GET /api/v1/rag/collections`

---

### 🔲 Phase 13 — Production Infrastructure

- [ ] Alembic migrations
- [ ] Kubernetes manifests
- [ ] Helm chart
- [ ] OpenTelemetry tracing
- [ ] Prometheus metrics
- [ ] Grafana dashboard
- [ ] `optillm` CLI
- [ ] Next.js dashboard
- [ ] HPA configuration
- [ ] Python SDK package

---

### 🔲 Phase 14 — Enterprise

- [ ] Multi-tenant namespace isolation
- [ ] RBAC (3 roles)
- [ ] API key management UI
- [ ] Audit log
- [ ] Encryption at rest
- [ ] PII detection/redaction
- [ ] Plugin system
- [ ] SLA manager
- [ ] Reference plugin

---

## 10. Development Rules

These rules are non-negotiable. Pull requests that violate them are not merged.

### Architecture Rules

1. **Architecture first, code second.** Every new module or feature needs a design note before implementation begins.

2. **No business logic in endpoints.** Endpoints validate input, call one service method, return the result. Period.

3. **No gateway logic in providers.** Provider files only translate between the internal format and the provider's API. Zero routing, caching, or analytics code.

4. **One responsibility per module.** If the module name requires "and", it must be split.

5. **Fail loudly, degrade gracefully.** External dependencies must have a defined fallback on failure. The gateway must never crash due to a Redis, vector DB, or provider outage.

### Code Quality Rules

6. **Follow SOLID principles.** Especially Single Responsibility, Open/Closed, and Dependency Inversion.

7. **Prefer composition over inheritance.** Inherit from `BaseProvider` only where required.

8. **No magic.** No dynamic dispatch, `__getattr__` tricks, or metaclasses without documented justification. Code must be readable by a new contributor on day one.

9. **Type hints everywhere.** Every function parameter and return type annotated. `mypy` strict is the goal.

10. **No commented-out code.** Delete unused code. Git history exists for recovery.

### Testing Rules

11. **Every public function in `engine/` has a unit test.** No exceptions.

12. **Every endpoint has an integration test.** Use `httpx.AsyncClient` with a test database.

13. **External calls are mocked in tests.** No test calls a real LLM provider.

14. **Tests are fast.** Unit tests < 1ms each. Integration tests < 500ms each.

### Documentation Rules

15. **Every module has a module-level docstring.** What it does, what it does not do, design decisions.

16. **Every public function has a docstring.** Args, returns, exceptions.

17. **Update this roadmap when phases complete or change.** This document is the source of truth.

18. **Update README when the API changes.** README always reflects current state.

### Process Rules

19. **Never merge experimental code into `main`.** Feature branches only. Promote to main when stable, tested, and documented.

20. **One feature per pull request.** Focused, reviewable, revertable.

21. **All CI checks pass before merge.** Lint, type check, test — all green.

22. **Semantic versioning.** `MAJOR.MINOR.PATCH`. Breaking API changes increment MAJOR.

---

> **Note to contributors:** This document is the single source of truth for the project. Update it when phases change. If you are working on a feature not listed here, open a discussion first to ensure architectural alignment.

---

*OptiLLM — Optimization before generation.*
