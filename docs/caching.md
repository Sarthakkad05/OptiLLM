# Semantic Caching Architecture

OptiLLM features a 2-tier semantic vector cache that matches incoming prompts based on **meaning**, not just identical text strings.

---

## 1. How It Works

```
User Query: "What is the speed of light in vacuum?"
     │
     ▼
[Normalized 384-dimensional Embedding]
(`all-MiniLM-L6-v2` — ~15ms local inference)
     │
     ├──► [Tier 1: Distributed Redis Cache] (if REDIS_URL configured)
     │      Cosine vector index search
     │
     └──► [Tier 2: Local FAISS Vector Store] (persisted to disk)
            Exact Inner-Product (IndexFlatIP) search
```

When a user submits:
`"Tell me the velocity of light in a vacuum."`

The embedding produces a cosine similarity score of **0.93** against the cached entry. Since `0.93 >= 0.90` (default threshold), OptiLLM immediately returns the cached response in **< 10ms** at **$0.00 cost**.

---

## 2. Threshold Tuning Guide

Configured via `CACHE_SIMILARITY_THRESHOLD` in `.env` or per-request via `optillm.cache_threshold`:

| Threshold | Matching Precision | Recommended Use Case |
|:---:|---|---|
| **0.80 – 0.84** | High recall, loose paraphrase | Customer support FAQs, conversational chit-chat |
| **0.85 – 0.90** | **Balanced (Default)** | General documentation, search summarization, enterprise knowledge bots |
| **0.91 – 0.95** | High precision, strict match | Math calculations, code generation, medical diagnosis |

---

## 3. Multi-Tenant Isolation (Namespaces)

To prevent cross-tenant cache leakage, entries can be partitioned into distinct namespaces:

```python
# Client request with tenant isolation namespace
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Fetch Q3 earnings report"}],
    extra_body={
        "optillm": {
            "cache_namespace": "tenant_acme_corp",
            "ttl_seconds": 3600
        }
    }
)
```

A prompt in namespace `tenant_acme_corp` will **never** hit or leak data to a query in namespace `tenant_beta_inc`.

---

## 4. Cache Management Endpoints

```bash
# Check cache statistics & vector count
curl http://localhost:8000/api/v1/cache/info

# Pre-warm cache on startup
curl -X POST http://localhost:8000/api/v1/cache/warm

# Clear cache entries (globally or by namespace)
curl -X DELETE "http://localhost:8000/api/v1/cache/clear?namespace=tenant_acme_corp"
```
