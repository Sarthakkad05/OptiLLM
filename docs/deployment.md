# Production Deployment Guide

Guide for deploying OptiLLM to production environments using Docker Compose, Kubernetes, and cloud infrastructure.

---

## 1. Production Architecture

In production, OptiLLM runs stateless behind a load balancer, using PostgreSQL for relational tracking and Redis for distributed semantic caching.

```
                    Internet / Internal VPC
                              │
                              ▼
                     [Cloud Load Balancer]
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    [OptiLLM Worker 1]                  [OptiLLM Worker 2]
            │                                   │
            ├───────────────┬───────────────────┤
            ▼               ▼                   ▼
    [PostgreSQL DB]   [Redis Cluster]   [Provider APIs]
    (Logs & Analytics) (Shared Cache)  (OpenAI, Anthropic, etc.)
```

---

## 2. Docker Compose Deployment (validated HA setup)

The repo's [docker-compose.yml](../docker-compose.yml) runs the full stack: N `optillm` replicas behind an nginx load balancer, sharing one PostgreSQL and one Redis, plus Prometheus and Grafana. Only nginx is published on the host (`:8000`); the replicas just `expose` 8000, so they can be scaled freely.

```bash
cp .env.example .env            # add provider keys, set API_KEY_AUTH_ENABLED=true
docker compose up -d --build --scale optillm=2
curl localhost:8000/health
```

| Service | Role | Host port |
|---|---|---|
| `nginx` | Round-robin LB (re-resolves `optillm` via Docker DNS, so `--scale` just works) | 8000 |
| `optillm` (xN) | Gateway replicas, stateless | - |
| `db` | PostgreSQL 15: request logs, budgets, keys | 5432 |
| `redis` | Shared rate-limit / cache state | 6379 |
| `prometheus` / `grafana` | Metrics and the provisioned OptiLLM dashboard | 9090 / 3000 |

Notes:

- `.env` is **not** baked into the image (see `.dockerignore`). Configuration reaches containers through `env_file` / `environment` only.
- The embedding model is baked into the image at build time, so a replica starts in seconds without network access to HuggingFace.
- If your host already runs a Postgres on `5432`, change the `db` port mapping. The container-to-container connection (`db:5432`) is unaffected.

### Schema migrations

Every replica runs `alembic upgrade head` on startup (`app/main.py`).

- **Concurrent boot is safe.** `migrations/env.py` takes a Postgres advisory lock, so replicas starting together serialize: one migrates, the rest see the DB at head and continue. (Without the lock, two replicas racing on a fresh DB crashed with `UniqueViolation` on `alembic_version`; this reproduced 3/3 times before the fix and 0/3 after.)
- **Failure is fatal.** A failed migration stops the replica from starting rather than serving traffic against a mismatched schema.
- **Migrations are idempotent.** They check whether a table or column exists instead of swallowing errors. (On Postgres, one swallowed failure aborts the whole transaction, which breaks every later statement.)

**Recovering a pre-Alembic database.** A database whose tables were created outside Alembic (no `alembic_version` table) fails migration 001 with `DuplicateTable`. Mark it as already at the initial revision, then upgrade. Migrations 002+ only add what is missing:

```bash
docker compose run --rm --entrypoint python optillm -m alembic stamp 001_initial_schema
docker compose up -d
```

---

## HA Validation Results

Validated with [benchmarks/bench_concurrency.py](../benchmarks/bench_concurrency.py) against the stack above: 2 replicas, nginx, shared Postgres 15 + Redis 7.

**What this does and does not measure.** Provider keys were blank (mock mode, via `docker-compose.loadtest.yml`), so these numbers reflect gateway, database and cache overhead under concurrency, **not** provider latency or spend. Rate limiting was raised so the load generator wouldn't trip it. Environment: Docker Desktop on macOS, 8 CPUs / 4 GB allocated, load generator on the same machine, so treat absolute throughput as indicative, not a capacity guarantee.

| Requests | Concurrency | Success | Throughput | p50 | p95 | p99 | Replica split |
|---:|---:|---:|---:|---:|---:|---:|---|
| 300 | 30 | 300 / 300 | 23.9 req/s | 1.18 s | 2.20 s | 2.72 s | 153 / 147 |
| 1,000 | 50 | 1,000 / 1,000 | 25.3 req/s | 1.73 s | 3.94 s | 4.17 s | 516 / 484 |
| 2,000 | 100 | 2,000 / 2,000 | 25.0 req/s | 3.97 s | 7.16 s | 7.54 s | 1032 / 968 |

To reproduce:

```bash
docker compose -f docker-compose.yml -f docker-compose.loadtest.yml up -d --build --scale optillm=2
python benchmarks/bench_concurrency.py --url http://localhost:8000 --requests 1000 --concurrency 50
# row-count check: count of request_logs should grow by exactly --requests
docker exec optillm_db psql -U optillm -d optillmdb -tAc "select count(*) from request_logs"
```

What was checked, and the result:

- **No errors.** 3,300 requests, 0 failures, 0 replica restarts, 0 `ERROR` log lines.
- **Load is actually distributed.** Both replicas served roughly half of every run (confirmed via the `X-Upstream-Addr` header nginx adds).
- **Shared state stays consistent.** After each run the `request_logs` row count grew by exactly the number of requests: nothing dropped or double-counted with two processes writing concurrently.
- **No leaked DB connections.** Zero connections left `idle in transaction` after each run.

Throughput plateaus at about 25 req/s on this setup, and latency beyond that is queueing (concurrency / throughput: 100 / 25 ≈ 4 s, matching the p50). Add replicas (and CPU) to raise the ceiling; this was not measured beyond 2 replicas.

### Issues found and fixed during validation

These only surfaced under multi-replica load or a clean deploy:

1. **Connection-pool deadlock under load.** Handlers held a pooled DB connection across the slow provider `await`. Handlers run sync DB calls on the event loop, so once ~15 requests were in flight the loop blocked on pool checkout, and the requests holding connections could never resume to release them. The replica hung permanently (health checks still passed on the other replica, masking it). Fixed by committing before the provider call (`app/services/gateway.py`) and closing the session when the handler returns (`release_db_on_return` in `app/db/session.py`).
2. **Concurrent migration race** on first boot (see above).
3. **Migration 002** assumed the `api_keys` table existed and poisoned the Postgres transaction on the first swallowed error.
4. **`requirements.txt`** was missing `langchain-core` / `langgraph` and pinned a conflicting `pydantic`, so a clean `pip install` could not run the app.
5. **Image build:** full CUDA `torch` was being pulled (multi-GB, unused). CPU wheel is now installed first. Missing `.dockerignore` shipped the host `venv/`, `.git` and `.env` into the image (3.7 GB to 2.1 GB after fix).
6. **Slow first boot:** the embedding model downloaded on every start (about 70 s with two replicas), outlasting the health check. It is now baked into the image.

---

## 3. Kubernetes Deployment & Health Probes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: optillm-gateway
  labels:
    app: optillm
spec:
  replicas: 3
  selector:
    matchLabels:
      app: optillm
  template:
    metadata:
      labels:
        app: optillm
    spec:
      containers:
      - name: optillm
        image: optillm:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: optillm-secrets
        - configMapRef:
            name: optillm-config
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        # ── Kubernetes Probes ──────────────────────────────────────
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5
          timeoutSeconds: 3
```

---

## 4. Production Checklist

1. **Database:** Set `DATABASE_URL` to a managed PostgreSQL instance (e.g. AWS RDS, GCP Cloud SQL).
2. **Authentication:** Set `API_KEY_AUTH_ENABLED=true`.
3. **Cache Storage:** Ensure `REDIS_URL` points to a clustered Redis instance with eviction policy `volatile-lru`.
4. **Input Size Caps:** Verify `MAX_REQUEST_BYTES=1048576` (1MB cap protects from memory exhaust).
5. **Observability:** Connect `OTEL_EXPORTER_OTLP_ENDPOINT` to your Datadog, Grafana Tempo, or Jaeger collector.
