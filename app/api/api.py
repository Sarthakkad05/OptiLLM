from fastapi import APIRouter

from app.api.endpoints import analytics, health, providers, proxy

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(analytics.router, prefix="/api/v1")
api_router.include_router(providers.router, prefix="/api/v1/providers")
api_router.include_router(proxy.router)
