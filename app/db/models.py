from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func

from app.db.base_class import Base


class RequestLog(Base):
    """
    Stores every request that passes through the OptiLLM gateway.
    Used by the analytics engine and dashboard.
    """

    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)

    # Request info & tags
    model_requested = Column(String, nullable=False, index=True)
    model_used = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False, default="openai", index=True)
    tag = Column(String(100), nullable=True, index=True)  # Header: x-optillm-tag

    # Token tracking
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    tokens_saved = Column(Integer, default=0)  # Via compression

    # Cost tracking
    cost_usd = Column(Float, default=0.0)  # Actual cost incurred
    savings_usd = Column(Float, default=0.0)  # Savings (cache + compression + routing)

    # Optimisation flags
    cache_hit = Column(Boolean, default=False, index=True)
    compressed = Column(Boolean, default=False, index=True)
    routed = Column(
        Boolean, default=False, index=True
    )  # True if routed to cheaper model

    # Performance
    latency_ms = Column(Integer, default=0)

    # Snippet for dashboard display (first 200 chars of prompt)
    prompt_snippet = Column(String(200), nullable=True)


class CacheEntry(Base):
    """
    Stores cached LLM responses mapped to their FAISS vector index ID.
    The FAISS index holds the embedding; this table holds the response text.
    """

    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True, index=True)
    faiss_index_id = Column(Integer, unique=True, nullable=False, index=True)

    # The raw prompt text — used to re-embed and rebuild FAISS on startup
    prompt_text = Column(Text, nullable=False, default="")

    response_text = Column(Text, nullable=False)
    model = Column(String, nullable=False)
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

    # TTL: if set, cache entry expires after this timestamp
    expires_at = Column(DateTime, nullable=True)


class KeyBudget(Base):
    """
    Per-API Key spend budget management (daily & monthly caps).
    """

    __tablename__ = "key_budgets"

    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String(100), unique=True, nullable=False, index=True)

    daily_budget_usd = Column(Float, default=10.0)
    monthly_budget_usd = Column(Float, default=100.0)

    daily_spent_usd = Column(Float, default=0.0)
    monthly_spent_usd = Column(Float, default=0.0)

    last_reset_day = Column(String(10), nullable=True)  # YYYY-MM-DD
    last_reset_month = Column(String(7), nullable=True)  # YYYY-MM
