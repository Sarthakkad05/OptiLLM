# OptiLLM — Interview Prep

---

## What is this project?

**One-liner:** OptiLLM is an LLM optimization gateway that sits between your application and any LLM provider (OpenAI, Gemini) and automatically reduces cost and latency through semantic caching, context compression, and intelligent model routing.

**The pitch:** Instead of paying $X for every GPT-4o call, you drop in a single `base_url` change and the gateway handles caching repeated questions, compressing bloated prompts, and routing simple tasks to cheaper models — transparently, with no code changes.

---

## Core Concepts

### 1. Semantic Cache
**Q: How does your caching work? Why not just use Redis with exact string matching?**

Exact matching would miss "What's your return policy?" vs "Tell me about returns" — they're the same question. We use sentence embeddings (all-MiniLM-L6-v2) to convert prompts into 384-dimension vectors and store them in a FAISS index. On every request, we find the nearest neighbor. If cosine similarity ≥ 0.95, it's a cache hit — we return the stored response in ~10ms at zero LLM cost.

**Q: What was a bug you found and fixed in the cache?**

The FAISS index persists to disk but the DB (SQLite) stores responses separately, both using sequential integer IDs. After a restart, these IDs can drift — FAISS returns ID 0 but the DB has no matching row. Every lookup was falling through to a miss even for identical prompts. I fixed it by storing the raw prompt text in the DB and rebuilding the FAISS index from DB embeddings on every startup.

**Q: What similarity threshold did you pick and why?**

0.95 cosine similarity. Lower than that and you risk false positives — returning a cached answer about "return policy" for a question about "return address." High-confidence cache hits only.

---

### 2. Context Compression
**Q: Why compress prompts? What's the tradeoff?**

LLM pricing is per token. A long system prompt with extra whitespace, repetition, or boilerplate in the middle of a long conversation can easily add 20-30% tokens you're paying for but don't need. We run a two-pass heuristic: clean whitespace/duplicates first, then center-truncate user messages (keeping start and end) and tail-truncate system prompts. The tradeoff is loss of context precision — so we only compress when the prompt exceeds a token threshold.

---

### 3. Model Router
**Q: How does the router decide which model to use?**

We score each request on 4 signals: token count, presence of complex keywords (e.g., "explain", "analyze", "compare"), conversation depth (turns), and code-related content. The score maps to three tiers:
- LOW (score ≤ 0): gpt-4o-mini / gemini-flash
- MEDIUM (score ≤ 2): gpt-4o-mini
- HIGH (score > 2): original requested model

Simple "What are your hours?" → cheap model. "Debug my OAuth integration and explain the RFC" → full model.

**Q: Can the routing be changed without restarting the server?**

Yes. There's a `POST /api/v1/router/config` endpoint that updates the routing table at runtime. You can switch LOW and MEDIUM tier models on the fly.

---

### 4. Provider Reliability
**Q: What happens if OpenAI goes down?**

The dispatcher has two layers of defense: exponential backoff retry (3 attempts: 1s, 2s, 4s) and automatic provider fallback. If OpenAI exhausts retries, it automatically tries Gemini with a comparable model, and vice versa. A response header `provider: openai-fallback` indicates a fallback occurred.

---

### 5. Architecture
**Q: Walk me through what happens on a single request.**

```
Client → FastAPI proxy endpoint
  → Cache check (FAISS similarity search)
    → HIT: return cached response immediately (~10ms)
    → MISS:
      → Compressor (reduce tokens if needed)
      → Router (select cheapest model for this complexity)
      → Provider Dispatcher (OpenAI or Gemini, with retry)
      → Insert response into cache
      → Log to DB (cost, latency, model used, savings)
  → Return enriched response with optillm_metadata
```

**Q: How is this different from LiteLLM or LangChain?**

LiteLLM is a unified provider interface — it doesn't cache or route. LangChain is a framework for building LLM apps. OptiLLM is specifically an optimization proxy layer: it's provider-agnostic, adds no SDK dependency to your app (just a `base_url` change), and the optimizations run transparently on every request.

---

### 6. Python SDK
**Q: How would someone integrate this into an existing project?**

```python
# Before
from openai import OpenAI
client = OpenAI(api_key="sk-...")

# After — zero other changes
from optillm_client import OptiLLM
client = OptiLLM(api_key="sk-...", gateway_url="http://localhost:8000")

# Same call, enriched response
response = client.chat.completions.create(model="gpt-4o", messages=[...])
print(response.optillm_metadata.cache_hit)    # True/False
print(response.optillm_metadata.savings_usd)  # e.g. 0.004200
```

The SDK wraps `openai.OpenAI`, overrides `base_url` to point to the gateway, and attaches metadata to every response. Existing code needs no other changes.

---

## System Design Questions

**Q: How would you scale this to handle 10,000 requests/second?**

- Replace SQLite with PostgreSQL + connection pooling
- Replace in-process FAISS with a dedicated vector DB (Qdrant, Pinecone, or Weaviate) that supports horizontal sharding
- Run multiple uvicorn workers behind a load balancer (nginx)
- Move cache inserts to a background async queue so they don't block the response path
- Add Redis for a fast pre-check before hitting FAISS (exact hash match)

**Q: How do you handle cache invalidation?**

Two mechanisms: TTL (each cache entry has an optional `expires_at`) and manual invalidation via `DELETE /api/v1/cache/clear`. For production, you'd tag cache entries by domain (e.g., "pricing", "policy") and invalidate by tag when underlying data changes.

**Q: What are the failure modes?**

- FAISS index corruption → startup sync rebuilds from DB
- DB down → requests still flow through, just skip caching (graceful degradation)
- Both providers down → returns 503 after all retries exhausted
- Embedding model OOM → falls back to skipping cache, logs error

---

## Numbers to Know

| Metric | Value |
|--------|-------|
| Cache hit latency | ~10–80ms |
| Cold LLM call (gpt-4o) | 1–4 seconds |
| Embedding model | all-MiniLM-L6-v2 (384-dim, ~5ms/query on CPU) |
| Similarity threshold | 0.95 cosine |
| Retry attempts | 3 (1s → 2s → 4s backoff) |
| gpt-4o vs gpt-4o-mini cost | ~33x cheaper on input tokens |

---

## Questions to Ask the Interviewer

- How do you currently handle LLM cost optimization at scale?
- Do you use streaming responses? (We support passthrough streaming)
- What's your current cache strategy for LLM responses?
