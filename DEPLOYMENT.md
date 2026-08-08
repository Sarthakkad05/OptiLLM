# OptiLLM — Deployment Guide

> This guide walks through deploying OptiLLM from a fresh clone to a running production environment.

---

## Prerequisites

- Docker 24+ and Docker Compose v2
- At least one LLM provider API key (OpenAI, Anthropic, or Gemini)
- 2GB RAM minimum (4GB recommended — embedding model requires ~500MB)
- Linux/macOS (Windows WSL2 supported)

---

## Quick Start (Local Development)

### 1. Clone & Configure

```bash
git clone https://github.com/Sarthakkad05/OptiLLM.git
cd OptiLLM

# Copy environment template
cp .env.example .env

# Edit .env — set at minimum one API key:
# OPENAI_API_KEY=sk-proj-your-key-here
nano .env
```

### 2. Start with Docker Compose

```bash
docker compose up --build
```

This starts:
- `optillm_api` — Gateway on port 8000
- `optillm_db` — PostgreSQL on port 5432
- `optillm_redis` — Redis on port 6379

### 3. Verify Health

```bash
# Liveness check
curl http://localhost:8000/health

# Readiness check (checks DB + providers)
curl http://localhost:8000/ready

# Prometheus metrics
curl http://localhost:8000/metrics | grep optillm_

# Browse API docs
open http://localhost:8000/docs
```

### 4. Make Your First Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, OptiLLM!"}]
  }'
```

### 5. Create Your First API Key

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app", "project": "production"}'
```

### 6. Run Smoke Tests

```bash
pip install httpx
python scripts/smoke_test.py
```

---

## Production Deployment (Linux VM)

### Option A: Docker Compose on a Single VM

#### 1. Configure Production Environment

```bash
cp .env.example .env
```

Edit `.env` for production:
```bash
APP_ENV=production
DATABASE_URL=postgresql://optillm:STRONG_PASSWORD@db:5432/optillmdb
REDIS_URL=redis://redis:6379/0

OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...

API_KEY_AUTH_ENABLED=true
OPTILLM_API_KEYS=sk-optillm-your-production-key

LOG_LEVEL=INFO
```

#### 2. Deploy

```bash
docker compose up -d --build

# Check all services healthy
docker compose ps

# View logs
docker compose logs -f api
```

#### 3. Verify

```bash
curl http://YOUR_SERVER_IP:8000/health
python scripts/smoke_test.py --url http://YOUR_SERVER_IP:8000
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | Environment: `development` / `production` |
| `DATABASE_URL` | `sqlite:///./optillm.db` | Database connection string |
| `REDIS_URL` | _(empty)_ | Redis URL for distributed cache |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key |
| `ANTHROPIC_API_KEY` | _(empty)_ | Anthropic API key |
| `GEMINI_API_KEY` | _(empty)_ | Google Gemini API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_PROVIDER` | `openai` | Default LLM provider |
| `DEFAULT_MODEL` | `gpt-4o` | Default model |
| `API_KEY_AUTH_ENABLED` | `false` | Enable Bearer token auth |
| `OPTILLM_API_KEYS` | `sk-optillm-dev-key` | Comma-separated valid API keys |
| `RATE_LIMIT_PER_MINUTE` | `60` | Requests per minute per client |
| `CACHE_SIMILARITY_THRESHOLD` | `0.90` | Semantic cache similarity threshold |
| `ROUTING_MODE` | `shadow` | `rule_based` / `ai` / `shadow` |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `3` | Failures before circuit opens |
| `REQUEST_TIMEOUT_SECONDS` | `60.0` | LLM request timeout |

---

## Kubernetes Deployment

Kubernetes manifests are in `deploy/k8s/`. Helm chart is in `deploy/helm/`.

```bash
# Using Helm
helm install optillm deploy/helm/ \
  --set openai.apiKey=sk-proj-... \
  --set database.url=postgresql://...
```

---

## Monitoring

### Prometheus + Grafana

Grafana dashboards are in `deploy/grafana/`.

Metrics exposed at `GET /metrics`:
- `optillm_http_requests_total` — Request counts by method/endpoint/status
- `optillm_http_request_duration_seconds` — Latency histogram
- `optillm_cache_hits_total` — Cache hit counters
- `optillm_cache_misses_total` — Cache miss counters
- `optillm_tokens_saved_total` — Tokens saved by optimization
- `optillm_cost_savings_usd_total` — USD savings
- `optillm_provider_calls_total` — Provider call counts

### Log Format

All logs are structured with:
```
2026-08-08 19:00:00 | INFO | optillm.gateway | [req_id] Request | model=gpt-4o-mini | tokens=120 | ns=default
```

---

## Database Migrations

```bash
# Apply migrations to fresh database
alembic upgrade head

# Check migration status
alembic current

# Create new migration
alembic revision --autogenerate -m "description"
```

---

## Troubleshooting

### Gateway returns 502
- Check provider API keys are set correctly
- Run `curl http://localhost:8000/api/v1/providers/status`
- Check circuit breaker state in provider status

### Slow cold start (~20 seconds)
- Normal — embedding model (all-MiniLM-L6-v2) loads on startup
- Docker image includes model weights after first build

### Cache not working
- Check `GET /api/v1/cache/stats`
- Ensure `REDIS_URL` is set for distributed cache
- FAISS in-memory cache works without Redis
