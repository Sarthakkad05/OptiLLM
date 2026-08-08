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
from app.services.provider_health import start_health_check_task, stop_health_check_task


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

    # Idempotent migration guard for local dev SQLite databases
    from sqlalchemy import text
    with engine.connect() as conn:
        for col_stmt in [
            "ALTER TABLE cache_entries ADD COLUMN expires_at DATETIME",
            "ALTER TABLE cache_entries ADD COLUMN tenant_id VARCHAR(100) DEFAULT 'default'",
            "ALTER TABLE request_logs ADD COLUMN tag VARCHAR(100)",
            "ALTER TABLE request_logs ADD COLUMN quality_score FLOAT",
            "ALTER TABLE request_logs ADD COLUMN correctness_score FLOAT",
            "ALTER TABLE request_logs ADD COLUMN relevance_score FLOAT",
            "ALTER TABLE request_logs ADD COLUMN completeness_score FLOAT",
            "ALTER TABLE request_logs ADD COLUMN hallucination_score FLOAT",
            "ALTER TABLE request_logs ADD COLUMN efficiency_score FLOAT",
            "ALTER TABLE request_logs ADD COLUMN tenant_id VARCHAR(100) DEFAULT 'default'",
            "ALTER TABLE key_budgets ADD COLUMN tenant_id VARCHAR(100) DEFAULT 'default'",
        ]:
            try:
                conn.execute(text(col_stmt))
                conn.commit()
            except Exception:
                pass

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

    # Start provider health check background task
    start_health_check_task()

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    stop_health_check_task()
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
cors_origins = [
    origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()
]
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/metrics", include_in_schema=True, tags=["Observability"])
def get_prometheus_metrics():
    """
    Exposes live Prometheus metrics for scraping by Prometheus server.
    """
    from fastapi.responses import Response

    from app.core.metrics import render_metrics

    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@app.get("/playground", include_in_schema=True, tags=["Playground"])
def get_playground_ui():
    """
    Renders the interactive OptiLLM Gateway API Playground UI.
    """
    from pathlib import Path
    from fastapi.responses import HTMLResponse

    html_path = Path(__file__).parent / "static" / "playground.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Playground UI file not found</h1>", status_code=404)


