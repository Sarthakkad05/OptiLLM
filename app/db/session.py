import functools
from typing import AsyncGenerator, Awaitable, Callable, Generator, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# ── Sync engine (used by startup, background tasks, Alembic, and tests) ──────────────
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    connect_args=_connect_args,
    **({"pool_size": 5, "max_overflow": 10} if not _is_sqlite else {}),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Async engine (used by async FastAPI endpoint handlers) ──────────────────────
def _build_async_url(sync_url: str) -> str:
    """Convert a sync SQLAlchemy URL to its async driver equivalent."""
    if sync_url.startswith("sqlite:///"):
        return sync_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if sync_url.startswith("postgresql+psycopg2://"):
        return sync_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    return sync_url  # Return as-is for unknown dialects


_async_url = _build_async_url(settings.DATABASE_URL)
_async_connect_args = {"check_same_thread": False} if _is_sqlite else {}

try:
    async_engine = create_async_engine(
        _async_url,
        pool_pre_ping=True,
        connect_args=_async_connect_args,
        **({"pool_size": 5, "max_overflow": 10} if not _is_sqlite else {}),
    )
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    _async_available = True
except Exception:
    # Async driver not installed (e.g. aiosqlite/asyncpg missing) — degrade gracefully
    async_engine = None  # type: ignore[assignment]
    AsyncSessionLocal = None  # type: ignore[assignment]
    _async_available = False


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a synchronous DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_T = TypeVar("_T")


def release_db_on_return(handler: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
    """
    Decorator for async endpoints that take a sync `db: Session = Depends(get_db)`.

    Closes the session (returning its pooled connection) as soon as the handler
    returns or raises. get_db's own teardown only runs later, after the
    response has gone back out through the middleware stack — and any await
    in between while still holding a connection can deadlock the replica:
    these handlers run sync DB calls on the event loop, so once the pool is
    exhausted the loop itself blocks on checkout, and the coroutines holding
    connections can never resume to release them. The session stays usable
    after close() (e.g. by a StreamingResponse generator), re-acquiring a
    connection on demand.
    """
    @functools.wraps(handler)
    async def wrapper(*args, **kwargs):
        try:
            return await handler(*args, **kwargs)
        finally:
            db = kwargs.get("db")
            if db is not None:
                db.close()

    return wrapper


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — yields an async DB session per request.

    Use this in async endpoint handlers for non-blocking DB access.
    Falls back to a sync session wrapped in a thread if async driver is unavailable.

    Usage:
        @router.get("/example")
        async def my_endpoint(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(MyModel))
    """
    if not _async_available or AsyncSessionLocal is None:
        # Graceful degradation: yield a sync session via thread executor
        import asyncio
        loop = asyncio.get_event_loop()
        db = SessionLocal()
        try:
            yield db  # type: ignore[misc]
        finally:
            await loop.run_in_executor(None, db.close)
        return

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
