# Quickstart Guide

Get OptiLLM up and running in under 5 minutes.

---

## 1. Prerequisites

- Python 3.11+
- Git
- (Optional) Docker & Docker Compose for production deployments

---

## 2. Local Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sarthakkad05/OptiLLM.git
cd OptiLLM
```

### Step 2: Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and set your provider keys (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

> **Note:** If no provider keys are configured, OptiLLM automatically operates in local **Mock Mode**, allowing you to test client integrations, routing, caching, and rate limiting with zero external API dependencies.

---

## 3. Run the Gateway

Start the FastAPI application with live-reload:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see startup logs:
```
INFO:     🚀 OptiLLM Gateway starting up...
INFO:     Environment : development
INFO:     Default provider: openai / model: gpt-4o
INFO:     ✅ Database schema up to date (alembic upgrade head).
INFO:     ✅ Embedding model and FAISS index ready.
INFO:     ✅ Cache sync complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 4. Test Your First Request

### Option A: Standard cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-optillm-dev-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is semantic caching?"}]
  }'
```

Response includes standard OpenAI choices plus `optillm_metadata`:
```json
{
  "id": "chatcmpl-...",
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 42,
    "total_tokens": 57
  },
  "optillm_metadata": {
    "cache_hit": false,
    "compressed": false,
    "routed": false,
    "model_requested": "gpt-4o",
    "model_used": "gpt-4o",
    "latency_ms": 340,
    "cost_usd": 0.00045,
    "savings_usd": 0.0
  }
}
```

Repeat the exact same cURL command or a semantically similar prompt:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-optillm-dev-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain what semantic caching is."}]
  }'
```
Notice:
- `latency_ms` drops to **< 15ms**!
- `cache_hit: true`
- `cost_usd: 0.0`
- `savings_usd: 0.00045`

---

## 5. Health Probes

Verify system health:
```bash
# Liveness probe:
curl http://localhost:8000/health

# Structured readiness probe with latencies:
curl http://localhost:8000/health/ready
```
