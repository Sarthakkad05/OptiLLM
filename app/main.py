from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.core.errors import http_exception_handler, unhandled_exception_handler
from app.core.logging import logger
from app.core.middleware import RequestBodyLimitMiddleware, RequestLoggingMiddleware
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

    # Run Alembic migrations to bring schema up to date.
    # A failed migration means the DB schema may not match what the ORM
    # models expect — continuing to boot and serve traffic against a
    # mismatched schema risks silent errors or data corruption, so this is
    # fatal rather than a logged warning (see docs/deployment.md's
    # HA validation section for the incident that motivated this: a stale
    # pre-alembic Postgres volume caused exactly this failure mode).
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(
            "Alembic migration failed — refusing to start: %s",
            result.stderr.strip() or result.stdout.strip(),
        )
        raise RuntimeError("Alembic migration failed; see logs above for details.")
    logger.info("✅ Database schema up to date (alembic upgrade head).")


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
    version="1.0.0",
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
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.MAX_REQUEST_BYTES)
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


@app.get("/admin", include_in_schema=True, tags=["Admin"])
def get_admin_ui():
    """
    Renders the OptiLLM Admin UI — LiteLLM-style dashboard for managing keys,
    models, analytics, providers, cache, and budgets.
    """
    from pathlib import Path
    from fastapi.responses import HTMLResponse

    html_path = Path(__file__).parent / "static" / "admin.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Admin UI file not found</h1>", status_code=404)


