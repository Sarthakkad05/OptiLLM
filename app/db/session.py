from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from app.core.config import settings

# SQLite requires check_same_thread=False for multi-threaded use (FastAPI).
# For Postgres/other DBs this arg is irrelevant and ignored.
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Validates connections before use
    connect_args=_connect_args,
    # SQLite doesn't support pool_size/max_overflow
    **({"pool_size": 5, "max_overflow": 10} if not _is_sqlite else {}),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
