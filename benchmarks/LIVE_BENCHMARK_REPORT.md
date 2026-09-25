# OptiLLM Live-Traffic Benchmark Report

> **Generated:** 2026-09-22 13:42 UTC  
> **Environment:** Real OpenAI API calls (gpt-4o, gpt-4o-mini), real LLM-as-judge scoring.  
> **Measured spend for this run:** $0.0195 across 37 API calls.

This report exists to answer what `benchmarks/BENCHMARK_REPORT.md` (local/synthetic) cannot: when OptiLLM routes to a cheaper model or compresses context, does judged answer quality hold up against real provider responses? Scope is OpenAI-only for this pass.

---

## 1. Router Quality: Requested Model vs. Routed Model

Quality scored 0.0-1.0 by GPT-4o-mini acting as an LLM judge (correctness, relevance, completeness, conciseness, safety — see `app/evaluation/judge.py`). A negative delta means the routed (cheaper) model scored lower than the originally requested model on the same prompt.

| Complexity | Routed? | Routed Model | Baseline Quality (gpt-4o) | Routed Quality | Delta | Prompt |
|---|:---:|---|:---:|:---:|:---:|---|
| low | Yes | `gpt-4o-mini` | 0.96 | 0.96 | +0.00 | What is the capital of Australia? |
| low | Yes | `gpt-4o-mini` | 1.00 | 1.00 | +0.00 | How many centimeters in 5 inches? |
| low | Yes | `gpt-4o-mini` | 1.00 | 1.00 | +0.00 | Define photosynthesis in simple terms. |
| low | Yes | `gpt-4o-mini` | 0.94 | 0.94 | +0.00 | Summarize the economic consequences of high inflat |
| low | Yes | `gpt-4o-mini` | 1.00 | 1.00 | +0.00 | Write a Python function to check if a binary tree  |
| medium | Yes | `gpt-4o-mini` | 0.83 | 0.82 | -0.01 | Design a distributed multi-datacenter consensus pr |

**Average quality delta on routed prompts:** -0.0025 (quality held up).

**Rule-based router misclassification observed this run:**

- Expected `high`, rule-based router assessed `medium` for: "Design a distributed multi-datacenter consensus pr..." (full prompt: "Design a distributed multi-datacenter consensus protocol with fault tolerance against Byzantine nodes."). This happened consistently across all three runs of this benchmark. Notably, the console log shows `Shadow mode disagreement: Rule=medium vs AI=high (conf=0.68)` — the shadow-mode AI classifier correctly flagged this as `high` complexity, but shadow mode only logs the disagreement without acting on it. This is real evidence that the rule-based router underestimates at least some genuinely complex prompts, and a concrete argument for validating and promoting the AI router out of shadow mode (see Phase 2 of the project roadmap / `app/engine/router_trainer.py`).

---

## 2. Compression Quality: Smart-Compressed vs. Minimal Context

Both variants are judged against the **original, uncompressed conversation** — the question is whether the smart-compressed prompt still lets the model answer what the user actually asked.

| Conversation | Token Reduction | Uncompressed Quality | Compressed Quality | Delta |
|---|:---:|:---:|:---:|:---:|
| medium | 0.0% | 0.98 | 1.00 | +0.01 |
| large | 77.8% | 0.82 | 0.85 | +0.02 |

---

## 3. Direct (Uncached) Call Latency — Real Provider Round Trip

Measured over 5 real `gpt-4o-mini` calls with a short prompt, for comparison against the local semantic-cache hit latency (~8ms, see `benchmarks/BENCHMARK_REPORT.md`):

- **Average:** 1460 ms
- **P50:** 1492 ms
- **Min / Max:** 1207 ms / 1618 ms

---

## Known Limitations of This Run

- OpenAI only — Anthropic/Gemini/Groq/Mistral not yet validated live.
- Small sample sizes (a handful of prompts per category), chosen to bound cost — not statistically rigorous.
- The judge model (gpt-4o-mini) grading responses from gpt-4o-mini in the compression test introduces some same-model bias; a stronger/independent judge model would be a good follow-up.
- No concurrency — this measures single-request behavior, not production load.
