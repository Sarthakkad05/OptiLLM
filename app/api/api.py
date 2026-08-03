from fastapi import APIRouter

from app.api.endpoints import (
    analytics,
    budgets,
    compression_preview,
    enterprise,
    evaluation,
    health,
    prompts,
    providers,
    proxy,
    rag,
    router_config,
    tools,
    workflows,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(analytics.router, prefix="/api/v1")
api_router.include_router(compression_preview.router, prefix="/api/v1")
api_router.include_router(providers.router, prefix="/api/v1/providers")
api_router.include_router(router_config.router, prefix="/api/v1")
api_router.include_router(evaluation.router, prefix="/api/v1")
api_router.include_router(rag.router, prefix="/api/v1")
api_router.include_router(enterprise.router, prefix="/api/v1")
api_router.include_router(budgets.router, prefix="/api/v1/budgets")

api_router.include_router(prompts.router, prefix="/api/v1/prompts")
api_router.include_router(workflows.router, prefix="/api/v1/workflows")
api_router.include_router(tools.router, prefix="/api/v1/tools")
api_router.include_router(proxy.router)



