from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import logger
from app.api.api import api_router
from app.engine.embedding import load_model
from app.engine.faiss_store import load_index

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Cost Optimization Layer — sits between your app and LLM providers.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(api_router)


@app.on_event("startup")
async def on_startup():
    logger.info("🚀 OptiLLM Gateway starting up...")
    logger.info("Environment : %s", settings.APP_ENV)
    logger.info("Default provider: %s / model: %s", settings.DEFAULT_PROVIDER, settings.DEFAULT_MODEL)
    # Preload embedding model and FAISS index to avoid cold-start on first request
    load_model()
    load_index()
    logger.info("✅ Embedding model and FAISS index ready.")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 OptiLLM Gateway shutting down.")

