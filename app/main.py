from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.core.errors import http_exception_handler, unhandled_exception_handler
from app.core.logging import logger
from app.core.middleware import RequestLoggingMiddleware
from app.db.base import Base  # noqa: F401 — ensures all models are registered
from app.db.session import SessionLocal, engine
from app.engine.cache import sync_cache_on_startup
from app.engine.embedding import load_model
from app.engine.faiss_store import load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle of the gateway."""
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("🚀 OptiLLM Gateway starting up...")
    logger.info("Environment : %s", settings.APP_ENV)
    logger.info(
        "Default provider: %s / model: %s",
        settings.DEFAULT_PROVIDER,
        settings.DEFAULT_MODEL,
    )

    # Ensure all DB tables exist (idempotent — safe to run on every start)
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables ready.")

    # Preload embedding model and FAISS index to avoid cold-start on first request
    load_model()
    load_index()
    logger.info("✅ Embedding model and FAISS index ready.")

    # Sync FAISS index with DB to fix ID-drift across restarts
    logger.info("🔄 Syncing FAISS index with DB cache entries...")
    db = SessionLocal()
    try:
        sync_cache_on_startup(db)
    finally:
        db.close()
    logger.info("✅ Cache sync complete.")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("🛑 OptiLLM Gateway shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Gateway — semantic caching, context compression, and intelligent model routing.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Exception handlers for RFC 7807 problem details
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Middlewares
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
