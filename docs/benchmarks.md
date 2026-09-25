# OptiLLM — Performance & Cost-Efficiency Benchmarks

OptiLLM is engineered as a zero-friction, low-latency AI Gateway that transparently sits between applications and LLM providers, aiming to add only single-digit-millisecond overhead while cutting API token costs.

## ⚠️ Validation Status

**Everything on this page below is a local/synthetic benchmark**, run via `benchmarks/generate_report.py` against an in-memory FAISS index, SQLite, and mocked/fixed prompt sets — **not** live provider traffic, not production load, and not independently audited. It measures the pipeline's *own processing overhead* accurately (that part doesn't need live traffic to be true), but hit rates, down-routing rates, and especially the cost-savings figures in §5 are **illustrative projections based on list prices**, not observed outcomes.

A first live-traffic validation pass now exists — see [`benchmarks/LIVE_BENCHMARK_REPORT.md`](../benchmarks/LIVE_BENCHMARK_REPORT.md), produced by `benchmarks/bench_live.py` against real OpenAI API calls with real LLM-as-judge quality scoring. Headline findings (OpenAI-only, small sample, see that report's own limitations section):

- On the router-quality test, down-routed responses held up in quality (average delta -0.0025 across routed prompts) — but the run also caught a real rule-based router **misclassification**: a genuinely complex prompt was assessed as `medium` and down-routed, while the shadow-mode AI classifier correctly flagged it as `high` (logged, not acted on). That's concrete evidence for prioritizing the AI router's promotion out of shadow mode.
- Context compression showed no meaningful quality loss on this run's two conversations (deltas of +0.01 and +0.02), though sample size is too small to generalize.
- Real uncached OpenAI latency measured ~1.5s average — versus the ~8ms local cache-hit latency measured below, which is the number that actually matters for the cache's value proposition.

Cross-provider (Anthropic/Gemini/Groq/Mistral) live validation and load-tested concurrency numbers are still open — see the project roadmap. Until those land, treat everything below this note as directional, not a guarantee.

---

## 1. Executive Performance Summary (local, single-process)

| Optimization Layer | Benchmark Metric | Observed Performance | Baseline (reference figures, not measured here) |
|---|---|---|---|
| **Semantic Cache** | Cache Hit Response Latency | **8.36 ms** | Typical direct API call: several hundred ms to a few seconds, depending on provider/model |
| **Semantic Cache** | Paraphrased Query Recall (@ 0.85) | **100.0%** *(synthetic paraphrase set, n small)* | Exact-match cache: 0% recall by definition |
| **Intelligent Router** | Classification Overhead (P50) | **1.60 ms** | — |
| **Intelligent Router** | Low-Cost Down-Routing Rate | **100.0%** *(synthetic prompt set)* | — |
| **Smart TF-IDF Compression** | Token Reduction (Conversational) | **68.8%** | Uncompressed: 0% savings by definition |
| **Smart TF-IDF Compression** | Processing Latency | **5.56 ms** | — |
| **End-to-End Cost Savings** | Enterprise Blended Workloads | **62.4% – 81.5%** *(list-price arithmetic, §5 — not an observed result)* | — |

---

## 2. Semantic Cache Benchmarks

OptiLLM's semantic cache utilizes FAISS cosine similarity vector search with normalized 384-dimensional embeddings (`all-MiniLM-L6-v2`) backed by Redis/SQLite persistence.

### Latency Comparison

```
Direct LLM Call (P50)  │████████████████████████████████████████ ~1,240 ms (reference figure, not measured here)
OptiLLM Exact Cache     │█ 8.36 ms (measured, local)
OptiLLM Semantic Cache  │█ 7.41 ms (measured, local)
```

The cache-hit latencies are real local measurements. The "Direct LLM Call" bar is an illustrative reference point (typical provider latency), not something this benchmark run measured — a live-traffic comparison against the same prompts is the planned next step.

### Hit Rate & Latency by Similarity Threshold

| Cosine Threshold ($\tau$) | Semantic Hit Rate | Avg Latency | P95 Latency | Best Suited For |
|---|---|---|---|---|
| **0.80** | **100.0%** | 16.06 ms | 29.27 ms | High recall (FAQ, Customer Support, Knowledge Base) |
| **0.85** *(Default)* | **100.0%** | 7.41 ms | 7.85 ms | General conversational tasks & summarization |
| **0.90** | **60.0%** | 7.05 ms | 7.47 ms | Precision tasks (Financial analysis, Documentation) |
| **0.95** | **0.0%** | 7.00 ms | 7.29 ms | Strict zero-drift (Code generation, Mathematical proofs) |

---

## 3. Intelligent Model Router Benchmarks

OptiLLM extracts linguistic complexity indicators (token count, structural depth, reasoning keywords, code blocks, JSON schemas) and predicts query complexity in single-digit milliseconds.

### Routing Latency Distribution

- **P50 Latency:** **1.60 ms**
- **P90 Latency:** **12.40 ms**
- **P95 Latency:** **238.59 ms** *(includes cold-start tree traversal)*
- **Overhead:** Adds $< 2\text{ms}$ on typical requests.

### Cost Savings via Down-Routing (illustrative, not measured)

When users send generic requests configured for flagship models (e.g. `gpt-4o` or `claude-3-5-sonnet`), OptiLLM dynamically evaluates query difficulty and can route simple lookups, translations, and boilerplate queries to `gpt-4o-mini` or `gemini-1.5-flash`. The routing *decisions* below are what the classifier actually produced on the synthetic prompt set; the cost/latency deltas are **arithmetic against published list prices**, not a live A/B measurement, and don't yet account for whether the cheaper model's answer quality held up (that check is the point of the planned live-traffic validation):

| Prompt Category | Requested Model | Routed Model | Latency Impact | Cost Reduction |
|---|---|---|---|---|
| Factual Q&A / Trivia | `gpt-4o` ($5.00/1M tok) | `gpt-4o-mini` ($0.15/1M tok) | -120 ms | **97.0%** |
| Language Translation | `gpt-4o` | `gpt-4o-mini` | -95 ms | **97.0%** |
| Summarization (Short) | `gpt-4o` | `gpt-4o-mini` | -110 ms | **97.0%** |
| Complex Code Generation | `gpt-4o` | `gpt-4o` (Preserved) | +1.6 ms | 0% (Quality preserved) |
| Multi-step Reasoning | `gpt-4o` | `gpt-4o` (Preserved) | +1.8 ms | 0% (Quality preserved) |

---

## 4. Context Compression Benchmarks

OptiLLM features three compression engines:
1. **Minimal:** Strips duplicate whitespace, markdown formatting artifacts, and control tokens.
2. **Smart (TF-IDF):** Computes cross-turn term importance scores to eliminate redundant sentences while retaining critical prompt context.
3. **Aggressive:** Aggressively truncates conversational history to the latest turns and key instructions.

### Token Reduction and Latency by Mode

| Mode | Tokens Before | Tokens After | Reduction % | Processing Latency |
|---|---|---|---|---|
| **Minimal** | 852 | 849 | **0.4%** | 0.62 ms |
| **Smart (TF-IDF)** | 852 | 266 | **68.8%** | 5.56 ms |
| **Aggressive** | 852 | 266 | **68.8%** | 0.95 ms |

---

## 5. Enterprise Cost Waterfall Simulation (hypothetical, not a measurement)

This is a **hypothetical illustration**, not data from any real deployment — no customer or benchmark run has actually processed 10M requests/month through OptiLLM. It exists to show how the three savings mechanisms compound *if* real-world hit/routing rates matched the synthetic benchmark numbers above; treat the percentages as a model to be replaced once live-traffic numbers exist, not a claim about actual savings.

Simulated blended workload (40% repetitive queries, 35% simple queries, 25% complex queries) at 10,000,000 monthly requests:

```
Unoptimized Direct Spend:                     $25,000.00 / month
  - 40% Semantic Cache Hits:                - $10,000.00 (Savings)
  - 35% Intelligent Down-routing:            -  $7,525.00 (Savings)
  - Context Compression (20% reduction):     -  $1,240.00 (Savings)
─────────────────────────────────────────────────────────────────
OptiLLM Net Monthly Provider Spend:            $6,235.00 / month
Net Enterprise Savings:                       $18,765.00 / month (75.1% Cost Cut)
```

---

## 6. How to Reproduce Benchmarks Locally

Run the automated reproducible benchmark suite using the OptiLLM CLI or Makefile:

```bash
# Run the full benchmark suite
make bench

# Or directly via Python
python -m benchmarks.generate_report
```

Benchmark outputs and breakdown tables will be generated in `benchmarks/BENCHMARK_REPORT.md`.
