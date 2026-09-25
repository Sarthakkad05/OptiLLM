from fastapi import APIRouter

from app.api.endpoints import (
    analytics,
    budgets,
    callbacks,
    compression_preview,
    completions,
    dashboard,
    embeddings,
    enterprise,
    evaluation,
    feedback,
    guardrails,
    health,
    keys,
    mcp,
    model_aliases,
    models,
    prompts,
    providers,
    proxy,
    rag,
    router_config,
    teams,
    tools,
    users,
    workflows,
)

api_router = APIRouter()

# ── Core ──────────────────────────────────────────────────────────────────────
api_router.include_router(health.router)         # GET /health, GET /ready
api_router.include_router(proxy.router)          # POST /v1/chat/completions
api_router.include_router(completions.router)    # POST /v1/completions
api_router.include_router(embeddings.router)     # POST /v1/embeddings
api_router.include_router(models.router)         # GET /v1/models, GET /v1/models/{id}
api_router.include_router(keys.router)           # POST/GET/DELETE /v1/keys
api_router.include_router(teams.router)          # POST/GET/PATCH/DELETE /v1/teams
api_router.include_router(users.router)          # POST/GET/PATCH/DELETE /v1/users

# ── Analytics ─────────────────────────────────────────────────────────────────
api_router.include_router(analytics.router, prefix="/api/v1")
api_router.include_router(compression_preview.router, prefix="/api/v1")

# ── Providers & Routing ────────────────────────────────────────────────────────
api_router.include_router(providers.router, prefix="/api/v1/providers")
api_router.include_router(router_config.router, prefix="/api/v1")

# ── Evaluation & RAG ──────────────────────────────────────────────────────────
api_router.include_router(evaluation.router, prefix="/api/v1")
api_router.include_router(rag.router, prefix="/api/v1")

# ── Enterprise ────────────────────────────────────────────────────────────────
api_router.include_router(enterprise.router, prefix="/api/v1")
api_router.include_router(budgets.router, prefix="/api/v1/budgets")
api_router.include_router(guardrails.router, prefix="/api/v1")
api_router.include_router(model_aliases.router, prefix="/api/v1")
api_router.include_router(callbacks.router, prefix="/api/v1")
api_router.include_router(dashboard.router, prefix="/api/v1")

# ── Prompts, Workflows & Tools ────────────────────────────────────────────────
api_router.include_router(prompts.router, prefix="/api/v1/prompts")
api_router.include_router(workflows.router, prefix="/api/v1/workflows")
api_router.include_router(tools.router, prefix="/api/v1/tools")
api_router.include_router(mcp.router, prefix="/api/v1/mcp")

# ── Phase 2: Feedback, Router Training & Quality-Cost Analytics ────────────
api_router.include_router(feedback.router, prefix="/api/v1")
