# OptiLLM

An AI Gateway that sits between your application and LLM providers — adding **semantic caching**, **context compression**, and **intelligent model routing** to every request, transparently.

```
Your App ──► OptiLLM Gateway ──► OpenAI / Gemini
                │
                ├── Semantic Cache   (skip the LLM call entirely)
                ├── Compressor       (reduce token count before sending)
                ├── Router           (use cheapest capable model)
                └── Analytics        (cost, savings, latency tracking)
```

---

## Features

| Feature | Description |
|---|---|
| **Semantic Cache** | Stores LLM responses as dense vector embeddings. Semantically similar follow-up questions are served from cache — no LLM call, zero cost. |
| **Context Compression** | Two-pass heuristic + token-aware truncation reduces prompt token count before sending to the provider. |
| **Model Routing** | Rule-based complexity analysis downgrades expensive models to cheaper ones when the task is simple enough. Never upgrades. |
| **Provider Abstraction** | Unified interface over OpenAI and Gemini. Automatic retry with exponential backoff + cross-provider fallback. |
| **Mock Mode** | No API keys? The full pipeline runs with simulated responses — useful for local development. |
| **Analytics API** | Every request is logged. Query cost, savings, cache hit rate, latency, and model distribution over time. |
| **OpenAI-Compatible** | Drop `http://localhost:8000` as your `base_url` in any OpenAI SDK call. Zero client changes. |

---

## Architecture

```
app/
├── api/
│   ├── api.py                  # Router registration
│   └── endpoints/
│       ├── health.py           # GET /health
│       ├── proxy.py            # POST /v1/chat/completions
│       └── analytics.py        # GET /api/v1/analytics + cache/routing endpoints
├── core/
│   ├── config.py               # Pydantic settings (env-driven)
│   └── logging.py              # Structured logging setup
├── db/
│   ├── models.py               # RequestLog, CacheEntry (SQLAlchemy)
│   └── session.py              # DB engine + session factory
├── engine/
│   ├── cache.py                # Semantic cache orchestration
│   ├── compressor.py           # Context compression pipeline
│   ├── embedding.py            # SentenceTransformer wrapper (singleton)
│   ├── faiss_store.py          # FAISS index management
│   └── router.py               # Rule-based model complexity routing
├── providers/
│   ├── dispatcher.py           # Provider selection, retry, fallback
│   ├── openai_client.py        # OpenAI Chat Completions client
│   └── gemini_client.py        # Google Gemini generateContent client
├── schemas/
│   ├── chat.py                 # Request / response Pydantic models
│   └── analytics.py            # Analytics response schemas
├── services/
│   ├── gateway.py              # Main request pipeline orchestrator
│   ├── analytics.py            # Analytics aggregation queries
│   ├── cost_estimator.py       # USD cost calculations
│   └── token_counter.py        # tiktoken-based token counting
└── main.py                     # FastAPI app + lifespan
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- At least one API key: `OPENAI_API_KEY` or `GEMINI_API_KEY` (or neither for mock mode)

### Local Development (SQLite)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd OptiLLM

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY or GEMINI_API_KEY
# DATABASE_URL defaults to sqlite:///./optillm.db if not set

# 5. Start the gateway
uvicorn app.main:app --reload
```

The gateway is now running at **http://localhost:8000**.

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Docker (Postgres)

```bash
# Copy and configure environment
cp .env.example .env

# Start Postgres + gateway
docker compose up --build
```

---

## Configuration

All settings are read from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./optillm.db` | SQLAlchemy database URL |
| `OPENAI_API_KEY` | `""` | OpenAI API key |
| `GEMINI_API_KEY` | `""` | Google Gemini API key |
| `DEFAULT_PROVIDER` | `openai` | Fallback provider when none is specified |
| `DEFAULT_MODEL` | `gpt-4o` | Fallback model |
| `FAISS_INDEX_PATH` | `faiss_store/index.faiss` | Path for persisted FAISS vector index |
| `APP_ENV` | `development` | Environment label |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible endpoint. Point any OpenAI SDK client at `http://localhost:8000` with no other changes.

**Request body** (OpenAI format + optional `optillm` block):

```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Explain neural networks"}],
  "temperature": 0.7,
  "optillm": {
    "bypass_cache": false,
    "bypass_compression": false,
    "bypass_routing": false
  }
}
```

**Response** (OpenAI format + `optillm_metadata`):

```json
{
  "id": "chatcmpl-abc123",
  "model": "gpt-4o-mini",
  "choices": [...],
  "usage": {"prompt_tokens": 42, "completion_tokens": 180, "total_tokens": 222},
  "optillm_metadata": {
    "cache_hit": false,
    "compressed": false,
    "routed": true,
    "model_requested": "gpt-4o",
    "model_used": "gpt-4o-mini",
    "latency_ms": 843,
    "cost_usd": 0.000114,
    "savings_usd": 0.00243,
    "tokens_saved": 0,
    "routing_reason": "Task complexity=low — downgraded from gpt-4o to gpt-4o-mini.",
    "complexity": "low"
  }
}
```

### `GET /health`

```json
{"status": "ok", "service": "optillm-gateway", "database": "connected"}
```

### `GET /api/v1/analytics`

Returns aggregated KPIs, cost-over-time series, model distribution, and the 50 most recent request logs.

### `GET /api/v1/cache/stats`

Returns semantic cache health: total entries, expired entries, FAISS vector count, similarity threshold.

### `DELETE /api/v1/cache/clear`

Wipes all cached responses and resets the FAISS index.

### `GET /api/v1/router/config`

Returns the current routing table (which model is used per complexity tier).

### `POST /api/v1/router/config`

Update the routing table at runtime (no restart required).

```json
{"low_model": "gemini-2.0-flash", "medium_model": "gpt-4o-mini"}
```

### `GET /api/v1/compression/stats`

Returns aggregated token savings from context compression.

### `GET /api/v1/routing/stats`

Returns model routing decisions, routing rate, and savings breakdown.

---

## Using with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
)

# The response is a standard ChatCompletion object
print(response.choices[0].message.content)
```

OptiLLM handles caching, compression, and routing transparently.

---

## Model Routing Logic

The router analyzes each request across four dimensions:

| Signal | Weight |
|---|---|
| Token count | Low: 0–79 tokens (+0), Medium: 80–399 (+1), High: 400+ (+2) |
| Keyword complexity | Simple keywords (−1), neutral (0), complex keywords (+2) |
| Conversation depth | Single turn (0), 2–4 turns (+1), 5+ turns (+2) |
| Code content | No code (0), code detected (+2) |

**Score → Complexity → Model:**

| Score | Complexity | Default Model |
|---|---|---|
| ≤ 0 | LOW | `gemini-2.0-flash` |
| 1–2 | MEDIUM | `gpt-4o-mini` |
| ≥ 3 | HIGH | *(original model retained)* |

The router **only downgrades** — it never routes to a more expensive model than the one requested.

---

## Provider Support

| Provider | Models |
|---|---|
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` |
| **Google Gemini** | `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.0-pro` |

Both providers share the same normalised internal response format, keeping provider-specific logic isolated inside `app/providers/`.

---

## License

MIT
