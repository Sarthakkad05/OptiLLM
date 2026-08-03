from app.db.base_class import Base  # noqa: F401

# All model classes must be imported here so Base.metadata.create_all auto-detects them.
from app.db.models import (  # noqa: F401
    AuditLogEntry,
    CacheEntry,
    KeyBudget,
    RequestLog,
    Tenant,
    ToolAuditLog,
)
