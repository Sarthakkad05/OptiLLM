from fastapi import APIRouter

from app.api.endpoints import analytics, budgets, health, prompts, providers, proxy

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(analytics.router, prefix="/api/v1")
api_router.include_router(providers.router, prefix="/api/v1/providers")
api_router.include_router(budgets.router, prefix="/api/v1/budgets")
api_router.include_router(prompts.router, prefix="/api/v1/prompts")
api_router.include_router(proxy.router)
