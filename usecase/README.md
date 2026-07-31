# ShopEasy Support Bot — OptiLLM Use Case

A production-realistic customer support chatbot that runs on top of the OptiLLM gateway.
Every message is automatically cached, compressed, and routed — savings are visible in real time.

## What This Demonstrates

| Feature | How it shows up |
|---------|----------------|
| Semantic Cache | Ask "What is your return policy?" → 1st call: 1400ms, $0.004. 2nd similar call: ~10ms, $0.000 |
| Model Routing | "Can I cancel?" → routed to `gpt-4o-mini`. "Debug my API integration" → stays on `gpt-4o` |
| Context Compression | Conversations with long history are compressed before sending |
| Live Metrics | Right-hand panel shows hit/miss, model used, latency, cost, savings per message |

## Quick Start

```bash
# 1. Start the OptiLLM gateway (from project root)
uvicorn app.main:app --reload --port 8000

# 2. Launch the support bot (from project root)
streamlit run usecase/support_bot/app.py --server.port 8502
```

Open http://localhost:8502

## Try This Demo Script

Send these messages in order to watch the metrics change:

1. **"What is your return policy?"** — Cache MISS (first time), LLM call, real cost
2. **"How do I return an item?"** — Cache HIT (similar intent), 10ms, $0
3. **"What time do you ship orders?"** — Simple question, ROUTED to cheap model
4. **"Can you explain the difference between Standard, Express, and Overnight shipping and when I should use each?"** — Complex, stays on GPT-4o
5. **"Tell me about returns"** — Cache HIT again, proves semantic matching (not exact string)

Watch the hit rate and total savings climb on the right panel.

## Environment

```
OPTILLM_URL=http://localhost:8000   # gateway address (default)
```
