from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import logger
from app.api.api import api_router
from app.engine.embedding import load_model
from app.engine.faiss_store import load_index
from app.engine.cache import sync_cache_on_startup
from app.db.session import engine, SessionLocal
from app.db.base import Base  # noqa: F401 — ensures all models are registered

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Cost Optimization Layer — sits between your app and LLM providers.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware to allow requests from the local test area
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
async def on_startup():
    logger.info("🚀 OptiLLM Gateway starting up...")
    logger.info("Environment : %s", settings.APP_ENV)
    logger.info("Default provider: %s / model: %s", settings.DEFAULT_PROVIDER, settings.DEFAULT_MODEL)

    # Auto-create all DB tables (safe — only creates if not existing)
    logger.info("📦 Ensuring database tables exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables ready.")

    # Preload embedding model and FAISS index to avoid cold-start on first request
    load_model()
    load_index()
    logger.info("✅ Embedding model and FAISS index ready.")

    # Sync FAISS index with DB to fix ID-drift bug across restarts
    logger.info("🔄 Syncing FAISS index with DB cache entries...")
    db = SessionLocal()
    try:
        sync_cache_on_startup(db)
    finally:
        db.close()
    logger.info("✅ Cache sync complete.")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 OptiLLM Gateway shutting down.")
