# Evaluation & Quality-Cost Analytics

OptiLLM features a dual-mode evaluation engine that continuously monitors LLM output quality and determines the optimal quality-per-dollar tradeoff for your workload.

---

## 1. Dual-Mode Evaluation Engine

Configured in `.env`:

```ini
EVAL_ENABLED=true                # Run heuristic quality evaluation on all requests
EVAL_LLM_ENABLED=true           # Enable real LLM-as-judge (sample-based)
EVAL_SAMPLE_RATE=0.05           # Run LLM judge on 5% of traffic
EVAL_LLM_MODEL=gpt-4o-mini      # Cost-effective, capable judge model
```

### Evaluation Modes

1. **Mode 1 — Heuristic Evaluation (100% of traffic):**
   - Free, zero-latency overhead.
   - Evaluates response length, structural coherence, vocabulary diversity, and formatting.

2. **Mode 2 — Real LLM-as-Judge (1%–10% sample):**
   - Calls `gpt-4o-mini` with a structured scoring prompt.
   - Generates calibrated float scores [0.0 – 1.0] across five core dimensions:
     - **`correctness`**: Factual precision and logical soundness.
     - **`relevance`**: Direct alignment with user instructions.
     - **`completeness`**: Thoroughness without omission of critical steps.
     - **`conciseness`**: Information density without superfluous filler.
     - **`safety`**: Compliance with safety boundaries.

---

## 2. Quality vs Cost Tradeoff Analytics

Endpoint: `GET /api/v1/analytics/quality-cost-tradeoff`

Answers the critical enterprise question: **"Which model gives me the highest quality per dollar for my specific production queries?"**

### Sample Response

```json
{
  "by_model": [
    {
      "model": "gpt-4o-mini",
      "avg_cost_per_request": 0.00015,
      "avg_quality_score": 0.88,
      "quality_per_dollar": 5866.7,
      "p99_latency_ms": 720
    },
    {
      "model": "gpt-4o",
      "avg_cost_per_request": 0.00280,
      "avg_quality_score": 0.93,
      "quality_per_dollar": 332.1,
      "p99_latency_ms": 2450
    }
  ],
  "recommendation": "For your current workload, gpt-4o-mini delivers 94.6% of gpt-4o quality at 5.3% of the cost."
}
```

---

## 3. Per-Request Feedback API

Applications can submit explicit user feedback on any completion:

```bash
POST /api/v1/feedback/{request_id}
Content-Type: application/json

{
  "rating": 1,          // 1 = Good, 0 = Neutral, -1 = Bad
  "issue": "slow",      // Optional: wrong | slow | incomplete | hallucinated | other
  "notes": "Response took too long to stream"
}
```

### Feeding the Learning Loop

When a user marks a request with `rating = -1` and `issue = "wrong"`:
1. OptiLLM logs a `RouterTrainingLabel` with ground-truth complexity `high`.
2. On the next retraining cycle, the router adjusts its weights so similar prompts are **not** down-routed to cheaper models.
3. Your routing accuracy improves automatically over time without manual tuning.
