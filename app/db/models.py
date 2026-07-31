from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, func
from app.db.base_class import Base


class RequestLog(Base):
    """
    Stores every request that passes through the OptiLLM gateway.
    Used by the analytics engine and dashboard.
    """
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)

    # Request info
    model_requested = Column(String, nullable=False)       # What the client asked for
    model_used = Column(String, nullable=False)             # What was actually used (after routing)
    provider = Column(String, nullable=False, default="openai")  # openai | gemini

    # Token tracking
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    tokens_saved = Column(Integer, default=0)               # Via compression

    # Cost tracking
    cost_usd = Column(Float, default=0.0)                   # Actual cost incurred
    savings_usd = Column(Float, default=0.0)                # Savings (cache + compression + routing)

    # Optimisation flags
    cache_hit = Column(Boolean, default=False)
    compressed = Column(Boolean, default=False)
    routed = Column(Boolean, default=False)                 # True if routed to cheaper model

    # Performance
    latency_ms = Column(Integer, default=0)

    # Snippet for dashboard display (first 200 chars of prompt)
    prompt_snippet = Column(String(200), nullable=True)


class CacheEntry(Base):
    """
    Stores cached LLM responses mapped to their FAISS vector index ID.
    The FAISS index holds the embedding; this table holds the response text.

    prompt_text is stored so the FAISS index can be rebuilt from DB on startup,
    preventing the FAISS↔DB ID drift that causes cache misses after restarts.
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
