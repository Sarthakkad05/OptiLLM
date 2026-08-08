# OptiLLM — Architecture Guide

## Overview

OptiLLM is a production-grade LLM Gateway that sits between your application and downstream AI providers. It makes LLM applications cheaper, faster, more reliable, and observable while exposing a simple OpenAI-compatible API.

```
Your Application
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    OptiLLM Gateway                        │
│                                                           │
│  ┌──────────┐   ┌────────────┐   ┌─────────────────────┐ │
│  │  Auth &  │   │  Request   │   │    Optimization      │ │
│  │ Rate Limit│──▶│  Pipeline  │──▶│    Engine            │ │
│  └──────────┘   └────────────┘   └─────────────────────┘ │
│                                           │               │
│              ┌────────────────────────────┤               │
│              ▼                            ▼               │
│  ┌─────────────────┐           ┌──────────────────────┐  │
│  │  Semantic Cache  │           │  Provider Dispatcher │  │
│  │  (FAISS + Redis) │           │  (LB + CB + Retry)   │  │
│  └─────────────────┘           └──────────────────────┘  │
│                                           │               │
└───────────────────────────────────────────┼───────────────┘
                                            ▼
                            ┌───────────────────────────┐
                            │  LLM Providers             │
                            │  OpenAI │ Anthropic │ Gemini│
                            │  Ollama (local)             │
                            └───────────────────────────┘
```

## Request Pipeline

Each request flows through the following pipeline:

```
1. API Key Authentication (optional, configurable)
2. Rate Limiting (sliding window, per client)
3. Request Validation (Pydantic schemas)
4. Budget Pre-flight Check (daily/monthly caps)
5. Semantic Cache Lookup
   ├─ Redis (distributed, with TTL)
   └─ FAISS (in-memory vector search, namespace-isolated)
6. Context Compression (if cache miss)
   ├─ Whitespace cleaning
   ├─ Deduplication
   └─ Center-truncation
7. Model Routing
   ├─ Rule-based complexity scoring
   └─ AI/ML classifier (shadow mode by default)
8. Provider Selection
   ├─ LoadBalancer (cost_optimized / round_robin / least_latency)
   └─ CircuitBreaker health check
9. Provider Call (with exponential retry)
10. Response returned to client
11. Cache Insert (async — FAISS + Redis + DB)
12. Cost & Savings Calculation
13. RequestLog Persistence
14. Prometheus Metrics Update
```

## Components

### Gateway Service (`app/services/gateway.py`)
The orchestration core. Coordinates cache, compressor, router, and dispatcher.

### Semantic Cache (`app/engine/cache.py`)
Two-layer cache:
- **Redis** — distributed, fast, namespace-prefixed, TTL-based
- **FAISS** — in-memory vector similarity search (384d all-MiniLM-L6-v2)
- **PostgreSQL** — persistent ground truth, used for FAISS rebuilds

Similarity threshold: `0.90` by default (configurable via `CACHE_SIMILARITY_THRESHOLD`).

### Compressor (`app/engine/compressor.py`)
Three compression modes:
- `minimal` — whitespace cleaning only
- `smart` — cleaning + deduplication
- `aggressive` — cleaning + dedup + center-truncation

System messages are always protected from truncation.

### Router (`app/engine/router.py`)
Complexity classification (LOW/MEDIUM/HIGH) determines model tier:
- **LOW** → cheap fast model (e.g., gpt-4o-mini)
- **MEDIUM** → balanced model (e.g., gpt-4o)
- **HIGH** → best model (e.g., gpt-4o, claude-3-5-sonnet)

Scoring factors: token count, code signals, multi-step keywords, depth signals.

Shadow mode runs AI classifier in parallel, logs disagreements without affecting routing.

### Provider Dispatcher (`app/providers/dispatcher.py`)
- 3 retry attempts with exponential backoff
- Automatic fallback to next healthy provider
- Circuit breaker protection per provider
- Mock mode (when no providers available)

### Circuit Breaker (`app/core/circuit_breaker.py`)
Standard 3-state machine:
- **CLOSED** — healthy, traffic flows normally
- **OPEN** — tripped after N consecutive failures
- **HALF_OPEN** — recovery probe after timeout

### Load Balancer (`app/providers/load_balancer.py`)
Strategies:
- `cost_optimized` — selects cheapest provider for the model
- `round_robin` — rotates sequentially
- `least_latency` — selects provider with lowest EMA latency

## Database Schema

| Table | Purpose |
|-------|---------|
| `request_logs` | Per-request audit: model, tokens, cost, savings, latency, quality |
| `cache_entries` | Semantic cache with FAISS IDs, TTL, namespace |
| `key_budgets` | Per-key daily/monthly spend limits |
| `tool_audit_logs` | MCP tool execution audit trail |
| `tenants` | Multi-tenant configuration |
| `audit_log_entries` | SHA256-chained immutable audit trail |

## Advanced Features

### RAG Pipeline (`app/rag/`)
- Document ingestion with chunking (recursive/sentence/fixed-size)
- Qdrant vector store (`:memory:` for dev, server URL for prod)
- Similarity-based document retrieval

### Evaluation Engine (`app/evaluation/`)
Heuristic 5-dimension judge:
- **Correctness** (30%) — structural format alignment
- **Relevance** (25%) — prompt-response word overlap
- **Completeness** (20%) — response length scoring
- **Conciseness** (15%) — repetition detection
- **Safety** (10%) — unsafe pattern detection

### MCP Router (`app/core/mcp_router.py`)
Model Context Protocol tool proxy with:
- Tool output caching (reuses expensive tool results)
- Audit logging for every tool execution
- Built-in system tools + extensible external servers

### Workflow Engine (`app/core/workflow_engine.py`)
LangGraph-compatible DAG-based workflow execution for multi-step AI pipelines.
