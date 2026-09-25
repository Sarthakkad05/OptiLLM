# OptiLLM — Deep-Dive Technical Notes

> Every claim here is sourced directly from the actual code. File references are exact.

---

## 1. Semantic Caching — Similarity Threshold Logic

### What the threshold does
The similarity threshold (`τ`) is a **minimum cosine similarity score** that a cached embedding must exceed before we consider it a "hit" and return the stored response.

```
New query → embed → cosine_similarity(query_vec, cached_vec) ≥ τ → return cached response
                                                               < τ → call LLM
```

### Where it lives in code
```python
# app/engine/cache.py — line 30
SIMILARITY_THRESHOLD = settings.CACHE_SIMILARITY_THRESHOLD  # default 0.85

# check_cache() uses it like this:
threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
```

### The four threshold operating points (from benchmarks)

| τ (threshold) | Semantic Hit Rate | Avg Latency | Use Case |
|---|---|---|---|
| **0.80** | 100% | 16 ms | FAQ, Customer Support, Knowledge Base |
| **0.85** *(default)* | 100% | 7.4 ms | General conversational tasks |
| **0.90** | 60% | 7.1 ms | Financial analysis, precise docs |
| **0.95** | 0% | 7.0 ms | Code gen, math proofs — never hits |

### The cosine similarity math
The embeddings are **L2-normalized** before being stored. After normalization:

```
cosine_similarity(A, B) = A · B  (inner product of unit vectors = cosine)
```

This is why `IndexFlatIP` (Inner Product) behaves like cosine similarity — the normalization is done by `generate_embedding()`:
```python
# app/engine/embedding.py — line 49
embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
```

### How threshold interacts with Redis vs FAISS
- **Redis**: `lookup_redis_cache()` manually computes cosine similarity in Python for every key under the namespace. It returns the **best scoring entry ≥ threshold** (not just any match).
- **FAISS**: `faiss_store.search()` returns raw inner-product distances. Score is then compared against threshold in `check_cache()`:
```python
score = float(distances[0][0])
if faiss_id == -1 or score < threshold:
    return None  # MISS
```

### How to reason about threshold in an interview
- Lower τ → more hits → more savings → higher risk of returning a semantically wrong answer
- Higher τ → safer answers → fewer hits → less cost benefit
- The "right" τ depends on the application's tolerance for semantic drift. Strict domains (legal, medical, code) need ≥ 0.90; conversational domains do fine at 0.85.
- Per-request override: `check_cache(messages, db, similarity_threshold=0.92)` — useful when one endpoint needs stricter caching than the default.

---

## 2. Cache Invalidation Strategy

### The three invalidation mechanisms

#### 2a. TTL-Based Expiry (time-based)
Every cache entry has an optional `expires_at` column:
```python
# app/db/models.py — CacheEntry
expires_at = Column(DateTime, nullable=True)

# app/engine/cache.py — insert_cache()
ttl = ttl_seconds if ttl_seconds is not None else settings.CACHE_TTL_SECONDS
expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
```

On a FAISS hit, expiry is checked before returning:
```python
# check_cache() — lines 146-149
if entry.expires_at is not None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if entry.expires_at < now:
        return None  # expired
```

Redis handles its own TTL natively:
```python
client.setex(key, ttl, json.dumps(data))  # Redis key auto-expires
```

#### 2b. Full Manual Clear (admin-triggered)
```python
# app/engine/cache.py — clear_cache()
db.query(CacheEntry).delete()   # wipe DB
faiss_store.reset_index()       # wipe in-memory index + delete .faiss file
clear_redis_cache(namespace)    # delete all optillm:{ns}:cache:* keys
```

Triggered via `DELETE /api/v1/cache/clear` or `python -m optillm_client.cli cache clear`.

#### 2c. Namespace Isolation (logical invalidation)
Multi-tenant isolation acts as **scoped invalidation**: each tenant's cache is completely separate. Clearing `namespace=tenantA` doesn't touch `tenantB`.

### What's NOT implemented (known gap)
There is **no event-driven / write-through invalidation**. If an LLM provider's knowledge cutoff is updated or a document changes, the cache will continue serving stale responses until TTL expires or manual clear. This is a known trade-off: stale cache is acceptable for factual Q&A, unacceptable for live data queries.

---

## 3. FAISS Index Type (Flat / IVF / HNSW) & Rationale

### What's implemented: `IndexFlatIP`
```python
# app/engine/faiss_store.py — line 39
def _create_index() -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(VECTOR_DIM)  # 384 dimensions
```

The code comment explicitly documents the decision:
```
# Index type: IndexFlatIP (Inner Product)
#   - Exact nearest-neighbor search (no approximation)
#   - Works as cosine similarity when vectors are L2-normalized
#   - Appropriate for MVP scale (< 100k entries)
```

### Comparing the three FAISS index types

| Index | Algorithm | Accuracy | Speed | Memory | Right Scale |
|---|---|---|---|---|---|
| **IndexFlatIP** | Brute-force inner product | **Exact** | O(n) linear scan | Low | < 100K vectors |
| **IndexIVFFlat** | Clusters + scan within cluster | Near-exact | O(n/nlist) | Medium | 100K–10M vectors |
| **IndexHNSWFlat** | Navigable small world graph | Approximate | O(log n) | High | Any scale |

### Why Flat is the right choice here
1. **Scale**: A production LLM cache rarely exceeds 50K–100K distinct semantic queries. At 384 dims × 100K vectors × 4 bytes = **~147MB** — easily fits in memory.
2. **Exactness matters**: A cache miss sent to the wrong answer is worse than a miss. With approximate indexes (IVF, HNSW), you can miss true matches or retrieve wrong ones.
3. **Latency**: Even a brute-force scan over 100K × 384-dim vectors takes ~7ms on CPU (confirmed in benchmarks). That's well within the 8.36ms average cache hit latency.
4. **Simplicity**: No training step required. `IndexIVFFlat` requires calling `.train()` on a representative sample before use — adds operational complexity.

### When you'd upgrade to IVF or HNSW
If the cache grows past ~500K entries and latency degrades:
- **IVF (Inverted File)**: Partition vectors into `nlist` clusters. Search only the `nprobe` nearest clusters. Needs a training pass. Accuracy degrades if `nprobe` is too small.
- **HNSW**: Graph-based, no training needed, fast queries, but higher memory and cannot delete nodes (rebuild needed). Best for read-heavy, large-scale caches.

### The ID-drift problem and the fix
`IndexFlatIP` assigns integer IDs sequentially (0, 1, 2...) but **does not support deletion**. If you delete a DB entry, the FAISS ID and DB row go out of sync on restart.

Fix: `rebuild_from_entries()` reconstructs the exact index from DB on every startup, **padding with zero vectors** for deleted entries to preserve ID alignment:
```python
while expected_id < faiss_id:
    pad = np.zeros((1, VECTOR_DIM), dtype=np.float32)
    new_index.add(pad)  # placeholder to maintain sequential IDs
    expected_id += 1
```

---

## 4. FAISS Scaling Behavior as Cache Grows

### Time complexity of search
`IndexFlatIP.search()` performs a **brute-force dot product** over every vector in the index:

- **Time**: O(n × d) where n = number of vectors, d = dimensions (384)
- **At 10K vectors**: ~0.6ms scan
- **At 100K vectors**: ~6ms scan (still within the 8.36ms budget)
- **At 1M vectors**: ~60ms — too slow, need to upgrade index type

### Memory usage
```
Memory = n × d × 4 bytes (float32)
  10K vectors  × 384 × 4 = ~15 MB
 100K vectors  × 384 × 4 = ~147 MB
   1M vectors  × 384 × 4 = ~1.47 GB
```

At 1M+ entries, memory pressure and latency both become problems simultaneously.

### What breaks first at scale
1. **Redis linear scan**: The Redis implementation in `lookup_redis_cache()` does `client.keys(pattern)` → iterates **every key** in Python and computes cosine similarity manually. This is O(n) Python code, far slower than FAISS's optimized C++ BLAS operations. Redis becomes the bottleneck long before FAISS does.

2. **FAISS disk save on every insert**: `save_index()` is called in `add()` — serializes the full index to disk every time a new entry is added. At high insert rates this becomes a bottleneck. Fix: batch writes or async save.

### Scaling path summary
```
< 50K entries   → IndexFlatIP + Redis (current)
50K–500K        → IndexIVFFlat (needs training) or HNSW
500K–10M        → Distributed FAISS / Milvus / Weaviate / Pinecone
> 10M           → Full vector database as a service
```

---

## 5. Sentence Transformers Model Choice & Embedding Dimension

### The model: `all-MiniLM-L6-v2`
```python
# app/engine/embedding.py
_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # app/engine/faiss_store.py
```

### Why this specific model

| Property | Value | Why It Matters |
|---|---|---|
| **Architecture** | MiniLM (distilled from larger BERT) | Fast inference |
| **Dimensions** | 384 | 4× smaller than BERT-base (768), same cosine quality |
| **Inference speed** | ~5ms/query on CPU | Documented in `embedding.py` comments |
| **Training data** | 1B+ sentence pairs (NLI + paraphrase datasets) | Strong semantic similarity |
| **License** | Apache 2.0 | Production safe |
| **Model size** | ~80MB | Baked into Docker image |

### Why 384 dimensions specifically
- 384d gives **strong paraphrase recall** without over-indexing on surface form
- 768d (BERT-base) doubles memory and search time with marginal accuracy gain for short-text similarity
- 1536d (OpenAI `text-embedding-ada-002`) is 4× the memory but requires an API call (adds latency + cost) — defeats the purpose of a local cache
- 384d is the sweet spot: fits 100K vectors in ~147MB, stays fast with IndexFlatIP

### Alternative models considered
| Model | Dims | Why NOT chosen |
|---|---|---|
| `text-embedding-ada-002` | 1536 | External API call, adds cost/latency |
| `all-mpnet-base-v2` | 768 | 2× memory, marginal accuracy gain |
| `paraphrase-MiniLM-L3-v2` | 384 | Lower quality, fewer params |
| `bge-small-en` | 384 | Good, but less battle-tested |

### Singleton pattern
The model is loaded **once at startup** and reused for all requests:
```python
# load_model() called in app/main.py lifespan
def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model
```

Loading SentenceTransformer is expensive (~1–2 seconds). The singleton avoids paying this on every request.

---

## 6. Embedding Latency vs Cache-Hit Savings Tradeoff

### The fundamental tension
Every cache lookup — whether it hits or misses — must **generate an embedding first**. You spend ~5ms on embedding even for a miss that still goes to the LLM.

```
Cache MISS cost:
  embedding (~5ms) + FAISS/Redis search (~7ms) + LLM call (~1,200ms)
  = ~1,212ms total
  
Cache HIT cost:
  embedding (~5ms) + FAISS/Redis search (~7ms) + DB fetch (~1ms)
  = ~13ms total
  → saves ~1,187ms AND the full token cost
```

### Why embedding overhead is worth it
At any non-trivial cache hit rate, the math strongly favors the overhead:

```
Hit rate 10%  → avg latency = 0.10 × 13ms + 0.90 × 1,212ms = 1,092ms
Hit rate 40%  → avg latency = 0.40 × 13ms + 0.60 × 1,212ms = 733ms
Hit rate 60%  → avg latency = 0.60 × 13ms + 0.40 × 1,212ms = 493ms

Without cache  → avg latency = 1,200ms always
```

Even a **10% cache hit rate** saves ~100ms average latency and eliminates 10% of token spend.

### The embedding latency itself
From `embedding.py` docstring: **~5ms/query on CPU**. This is fast because:
- MiniLM-L6 is a small (6-layer) distilled model
- `convert_to_numpy=True` avoids PyTorch tensor overhead
- Single text encoding (not batched) — justified because the cache path is on the hot request path

### Batch embedding (startup path)
`generate_embeddings_batch()` is only used during `sync_cache_on_startup()` — bulk re-indexing. Batch processing is 3–5× more efficient than sequential calls because the GPU/CPU processes a matrix of texts at once. Not used on per-request path because you only ever embed one query at a time.

### The case for caching embeddings themselves
A future optimization: cache the embedding for a given text (memoize). If the same exact text comes in twice, skip the 5ms encode. Not currently implemented — the current approach re-embeds on every request even for identical text.

---

## 7. Context Compression — Technique & Information-Loss Tradeoff

### The two-pass pipeline
```python
# app/engine/compressor.py
def compress(messages, model, max_tokens=2000, mode="smart"):
    # Pass 1: Heuristic cleaning
    cleaned = _clean_messages(messages)  
    cleaned, _ = deduplicate_messages(cleaned)
    
    # Pass 2: TF-IDF truncation (if still over budget)
    if original_tokens > effective_max_tokens:
        final_messages = _truncate_messages(cleaned, effective_max_tokens, model)
```

### Pass 1 — Heuristic Cleaning (always runs)
```python
text = re.sub(r"\n{3,}", "\n\n", text)      # 3+ newlines → 2
text = re.sub(r" {3,}", " ", text)           # 3+ spaces → 1
text = re.sub(r"([-_*]{3,}\n?){2,}", "---\n", text)  # repeated ---/___/***
# deduplicate consecutive identical lines
```
Zero information loss — these are pure noise tokens.

### Pass 2 — TF-IDF Sentence Ranking (runs only when > 2000 tokens)
```python
# _compress_text_tfidf()
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
tfidf_matrix = vectorizer.fit_transform(sentences)
sentence_scores = tfidf_matrix.sum(axis=1).A1  # row-sum = total TF-IDF weight
```

**Position multipliers** (prevents pure TF-IDF from discarding critical context):
```python
if idx == 0:
    pos_multiplier = 1.5   # first sentence always boosted (context setting)
elif idx == total_sents - 1:
    pos_multiplier = 1.3   # last sentence boosted (the actual query)
```

Sentences are then selected greedily by score until the token budget is exhausted, then **sorted back into original order** before returning.

### Why original order matters
LLMs are sensitive to chronological order of information. Returning sentences sorted by TF-IDF score (descending) would produce incoherent text. The re-sorting step ensures the compressed output reads naturally.

### The three modes and their tradeoffs

| Mode | Token Reduction | Information Risk | Use When |
|---|---|---|---|
| `minimal` | ~0.4% | Near zero | Cleaning only, preserve everything |
| `smart` | ~68.8% | Low–Medium | Default: TF-IDF picks important sentences |
| `aggressive` | ~68.8%+ | Medium–High | Max savings, lower token ceiling (1000) |

### Where information loss actually happens
1. **Middle-of-conversation context**: TF-IDF scores pure frequency. A rare but critical fact mentioned once in the middle of a long conversation may score lower than repeated filler.
2. **Code blocks**: Code is NOT semantically tokenizable by TF-IDF. A code block with `def`, `for`, `return` scores very differently than its actual importance. The code block detection in Pass 1 (`_score_by_code_content`) is for routing, not compression — compression doesn't specially protect code.
3. **Named entities**: "PostgreSQL advisory lock" has low TF-IDF across the conversation if only mentioned once but is critical. TF-IDF doesn't understand semantics.

### Key design rule: compress `messages_to_send`, not `messages`
```python
# The original messages list is NEVER modified.
# Compression only applies to what gets sent to the LLM.
# Cache lookup always uses original messages → correct embedding.
```

### Live validation result
From `benchmarks/bench_live.py` against real OpenAI API: quality delta of **+0.01 and +0.02** on tested conversations — compression actually slightly improved scores on those samples (possibly by removing noise). Sample size too small to generalize.

---

## 8. FastAPI + PostgreSQL Schema for Usage Analytics

### The central table: `request_logs`

Every request that passes through the gateway writes one row:

```python
# app/db/models.py — RequestLog
class RequestLog(Base):
    __tablename__ = "request_logs"
    
    id               = Column(Integer, primary_key=True, index=True)
    timestamp        = Column(DateTime, server_default=func.now(), index=True)
    
    # What was asked vs what ran
    model_requested  = Column(String, index=True)   # "gpt-4o" (from client)
    model_used       = Column(String, index=True)   # "gpt-4o-mini" (after routing)
    provider         = Column(String, index=True)   # "openai" | "gemini" | ...
    tag              = Column(String(100), index=True) # x-optillm-tag header
    
    # Token economics
    tokens_input     = Column(Integer, default=0)
    tokens_output    = Column(Integer, default=0)
    tokens_saved     = Column(Integer, default=0)   # via compression
    
    # Cost attribution
    cost_usd         = Column(Float, default=0.0)   # actual incurred
    savings_usd      = Column(Float, default=0.0)   # cache + compress + routing
    
    # Optimization flags (boolean — enables fast GROUP BY analytics)
    cache_hit        = Column(Boolean, index=True)
    compressed       = Column(Boolean, index=True)
    routed           = Column(Boolean, index=True)  # down-routed to cheaper model
    
    # ML router observability
    shadow_disagreement    = Column(Boolean, index=True)
    ai_predicted_complexity = Column(String(20))     # "low|medium|high"
    
    # Performance
    latency_ms       = Column(Integer, default=0)
    
    # Quality scores (from LLM-as-judge)
    quality_score      = Column(Float)
    correctness_score  = Column(Float)
    relevance_score    = Column(Float)
    completeness_score = Column(Float)
    hallucination_score = Column(Float)
    efficiency_score   = Column(Float)
    
    # Attribution
    tenant_id  = Column(String(100), index=True)
    team_id    = Column(String(100), index=True)
    user_id    = Column(String(100), index=True)
```

### The analytics queries this enables

```sql
-- Cost savings breakdown
SELECT 
  SUM(savings_usd) FILTER (WHERE cache_hit) AS cache_savings,
  SUM(savings_usd) FILTER (WHERE routed)    AS routing_savings,
  SUM(savings_usd) FILTER (WHERE compressed) AS compression_savings
FROM request_logs
WHERE timestamp > NOW() - INTERVAL '7 days';

-- Shadow disagreement rate
SELECT 
  COUNT(*) FILTER (WHERE shadow_disagreement) * 100.0 / COUNT(*) AS disagreement_rate_pct
FROM request_logs
WHERE timestamp > NOW() - INTERVAL '24 hours';

-- Provider latency by model
SELECT model_used, AVG(latency_ms), PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)
FROM request_logs
GROUP BY model_used;
```

### FastAPI endpoint pattern
```python
# Analytics queries are served from app/services/analytics.py
# Endpoints in app/api/endpoints/dashboard.py
# Session dependency injection:
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/analytics/summary")
def analytics_summary(db: Session = Depends(get_db)):
    return analytics_service.get_summary(db, days=7)
```

### Supporting tables in the schema

| Table | Purpose |
|---|---|
| `cache_entries` | Stores cached responses mapped to FAISS IDs + TTL |
| `key_budgets` | Daily/monthly spend caps per API key |
| `router_training_labels` | Labeled training data for the ML router |
| `router_training_runs` | Audit trail of each retraining run (accuracy before/after) |
| `request_feedback` | Per-request user ratings (1/-1/0) with issue categories |
| `organizations` / `teams` / `users` | Multi-tenancy hierarchy with per-level budgets |
| `audit_log_entries` | SHA256-chained immutable audit log |

---

## 9. Cost Estimation Logic per Model Provider

### The pricing table
```python
# app/services/cost_estimator.py — PRICING_TABLE
# Format: {model: (input_price_per_1M_tokens, output_price_per_1M_tokens)} USD

PRICING_TABLE = {
    "gpt-4o":           (5.00,  15.00),
    "gpt-4o-mini":      (0.15,   0.60),
    "gpt-4":            (30.00, 60.00),
    "gpt-3.5-turbo":    (0.50,   1.50),
    "gemini-1.5-pro":   (3.50,  10.50),
    "gemini-2.0-flash": (0.10,   0.40),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku":  (0.80,  4.00),
    "claude-3-opus":    (15.00, 75.00),
    "ollama":           (0.00,   0.00),  # local, no API cost
}
```

### Core cost formula
```python
def estimate_cost(model, tokens_input, tokens_output) -> float:
    input_price, output_price = get_model_pricing(model)
    cost = (tokens_input / 1_000_000) * input_price + \
           (tokens_output / 1_000_000) * output_price
    return round(cost, 8)
```

**Example**: `gpt-4o` call with 500 input tokens, 200 output tokens:
```
cost = (500/1,000,000) × $5.00 + (200/1,000,000) × $15.00
     = $0.0025 + $0.003
     = $0.0055 per request
```

### Three types of savings, each calculated separately

**Cache savings** (100% of the cost — no LLM call happened):
```python
def estimate_cache_savings(model, tokens_input, tokens_output):
    return estimate_cost(model, tokens_input, tokens_output)  # full cost
```

**Compression savings** (reduced input tokens):
```python
def estimate_compression_savings(model, original_tokens, compressed_tokens):
    tokens_saved = max(0, original_tokens - compressed_tokens)
    input_price, _ = get_model_pricing(model)
    return (tokens_saved / 1_000_000) * input_price
```

**Routing savings** (differential between expensive and cheap model):
```python
def estimate_routing_savings(original_model, routed_model, tokens_input, tokens_output):
    original_cost = estimate_cost(original_model, tokens_input, tokens_output)
    routed_cost   = estimate_cost(routed_model, tokens_input, tokens_output)
    return max(0.0, original_cost - routed_cost)
```

**Example — routing gpt-4o → gpt-4o-mini** (1000 input, 300 output tokens):
```
gpt-4o    : (1000/1M)×$5.00 + (300/1M)×$15.00 = $0.0050 + $0.0045 = $0.0095
gpt-4o-mini: (1000/1M)×$0.15 + (300/1M)×$0.60  = $0.00015 + $0.00018 = $0.00033
Savings   = $0.0095 - $0.00033 = $0.00917 (96.5% cost reduction)
```

### Fallback for unknown models
```python
_FALLBACK_PRICING = (5.00, 15.00)  # gpt-4o rates

def get_model_pricing(model):
    if model in PRICING_TABLE:
        return PRICING_TABLE[model]
    # Prefix/suffix match (handles version suffixes like "gpt-4o-2024-05-13")
    for key in PRICING_TABLE:
        if model.startswith(key) or key.startswith(model):
            return PRICING_TABLE[key]
    logger.warning("Unknown model '%s' — using fallback pricing", model)
    return _FALLBACK_PRICING
```

### Where costs are stored
Every request writes `cost_usd` and `savings_usd` to `request_logs`. The analytics dashboard aggregates these over time windows (daily, weekly, monthly) to show ROI.

---

## 10. What's Left to Build / Hardest Bugs So Far

### ⚠️ Known limitations (from `docs/routing.md §4.5`)

**1. Train/eval split contamination (unfixed)**
The `build_training_dataset()` function takes a positional 80/20 slice without shuffling or stratification. When explicit labels are a minority of the dataset, they all land in the training portion — the eval set contains only noisy inferred labels. The self-reported "eval accuracy" is meaningless. Fix: stratified shuffle split (`sklearn.model_selection.train_test_split(..., stratify=labels, shuffle=True)`).

**2. Redis linear scan**
`lookup_redis_cache()` calls `client.keys(pattern)` and loops over every key in Python. This is O(n) in Python — much slower than FAISS's C++ BLAS scan. At 10K+ Redis keys, this becomes the latency bottleneck. Fix: use Redis' native vector search (RedisSearch module with VECTOR field type) or migrate to a purpose-built vector DB.

**3. FAISS disk write on every insert**
`faiss_store.add()` calls `save_index()` synchronously after every vector insertion. Under high write throughput this serializes the full index file on every request. Fix: async/batched saves, or switch to FAISS's `IndexIDMap` with WAL-style persistence.

**4. No semantic cache invalidation on content updates**
If the underlying knowledge changes (e.g., a policy doc is updated), cached answers remain stale until TTL expires. Fix: event-driven invalidation tied to document version hashes.

**5. Missing cross-provider live benchmarks**
The live-traffic validation only covers OpenAI. Anthropic, Gemini, Groq, Mistral provider behaviors in production are unvalidated at scale.

### 🐛 Hardest bugs discovered during development

**Bug 1 — Silent duplicate route handler (most impactful)**
Two `POST /api/v1/router/train` handlers registered at the same path. FastAPI matched the unsafe one first on every call. The safety gate (evaluate before hotswap) was 100% unreachable through the API. Discovered only by tracing the route table and running a live request. Fixed by removing the duplicate. The corrupted model (16% accuracy) was recovered to 80% by triggering the real pipeline via HTTP.

**Bug 2 — FAISS ID drift on restart**
`IndexFlatIP` assigns sequential IDs (0,1,2...) and doesn't support deletion. After a cache clear + restart, FAISS IDs no longer matched DB rows, causing wrong responses to be returned for cache hits. Fixed with `rebuild_from_entries()` which pads zero vectors for deleted entries to maintain ID alignment.

**Bug 3 — Concurrent replica UniqueViolation on migration**
Two replicas starting simultaneously both attempted to create the `alembic_version` row on a fresh DB → `UniqueViolation`. Reproduced 3/3 times. Fixed with a PostgreSQL advisory lock in `migrations/env.py` that serializes the migration step across replicas.

**Bug 4 — Shadow disagreement logging was invisible**
The `route()` function computed `shadow_disagreement = True` and logged it but never persisted it. Without DB persistence, the data was lost after log rotation. Fixed by adding `shadow_disagreement` and `ai_predicted_complexity` columns via migration `004`, populating them in `gateway.py`, and exposing them via new endpoints. The first real disagreement captured: Byzantine consensus design classified as `medium` by rule-based router, `high` by AI router.

**Bug 5 — nginx "Request Header Or Cookie Too Large"**
Default nginx header buffer (1KB) was too small for accumulated browser cookies on the admin dashboard. Fixed by adding to `deploy/nginx/nginx.conf`:
```nginx
client_header_buffer_size 4k;
large_client_header_buffers 4 16k;
```

### 🗺️ What would be built next (roadmap)
1. **Fix train/eval split** — stratified shuffle, separate eval set from inferred labels
2. **Promote AI router from shadow → active** — enough disagreement data now exists to justify it
3. **Redis native vector search** — replace Python linear scan with RedisSearch `VECTOR` type
4. **Cross-provider live benchmarks** — Anthropic, Gemini, Groq, Mistral with real LLM-judge scoring
5. **Streaming support** — current API buffers full response before returning; SSE/chunked streaming for low-latency UX
6. **Async FAISS saves** — background task to persist index rather than blocking on every insert
7. **Semantic cache invalidation hooks** — version-aware cache clearing tied to document updates

---

*All code references verified against the live repository. File paths are relative to `/Users/sarthakkad/Desktop/projects/OptiLLM/`.*
