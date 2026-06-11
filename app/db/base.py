from app.db.base_class import Base  # noqa: F401

# All model classes must be imported here so Alembic can auto-detect them.
from app.db.models import RequestLog, CacheEntry  # noqa: F401

