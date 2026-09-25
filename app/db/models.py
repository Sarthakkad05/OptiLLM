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

    # Shadow-mode router observability (see app/engine/router.py route())
    shadow_disagreement = Column(
        Boolean, default=False, index=True
    )  # True if the AI router's predicted complexity differed from the rule-based one
    ai_predicted_complexity = Column(
        String(20), nullable=True
    )  # What the AI router would have chosen, when shadow/ai mode ran it

    # Performance
    latency_ms = Column(Integer, default=0)

    # Snippet for dashboard display (first 200 chars of prompt)
    prompt_snippet = Column(String(200), nullable=True)

    # Evaluation & Quality Engine (Phase 11)
    quality_score = Column(Float, nullable=True)
    correctness_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    hallucination_score = Column(Float, nullable=True)
    efficiency_score = Column(Float, nullable=True)

    # Multi-tenancy (Phase 14)
    tenant_id = Column(String(100), default="default", index=True)
    # Team attribution (Phase 2)
    team_id = Column(String(100), nullable=True, index=True)
    user_id = Column(String(100), nullable=True, index=True)


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
    tenant_id = Column(String(100), default="default", index=True)


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
    tenant_id = Column(String(100), default="default", index=True)


class ToolAuditLog(Base):
    """
    Audit log table recording tool execution traces, execution duration, and outcomes.
    """

    __tablename__ = "tool_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)

    tool_name = Column(String(100), nullable=False, index=True)
    arguments = Column(Text, nullable=True)  # JSON serialized arguments
    execution_time_ms = Column(Float, default=0.0)
    success = Column(Boolean, default=True, index=True)
    result_summary = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    tenant_id = Column(String(100), default="default", index=True)


class Tenant(Base):
    """
    Enterprise Tenant entity.
    """

    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    api_key = Column(String(200), nullable=False, unique=True)
    role = Column(String(50), default="developer", index=True)  # admin | developer | read_only
    sla_target_ms = Column(Float, default=500.0)
    created_at = Column(DateTime, server_default=func.now())


class AuditLogEntry(Base):
    """
    Immutable & tamper-evident enterprise audit log table.
    Uses cryptographic SHA256 chain linking (prev_hash -> payload_hash).
    """

    __tablename__ = "audit_log_entries"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    tenant_id = Column(String(100), default="default", index=True)
    actor = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False, index=True)
    resource = Column(String(200), nullable=False)
    payload_hash = Column(String(64), nullable=False)
    prev_hash = Column(String(64), nullable=False)
    status = Column(String(50), default="success")


# ── Phase 2: Multi-Tenancy Hierarchy ────────────────────────────────────────


class Organization(Base):
    """
    Top-level organizational unit.
    Organizations contain multiple Teams.
    """
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Team(Base):
    """
    Team entity within an Organization.
    Teams have their own budget, RPM/TPM limits, and set of API keys.
    """
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    org_id = Column(String(100), nullable=True, index=True)  # FK to organizations.org_id

    # Budget
    daily_budget_usd = Column(Float, default=100.0)
    monthly_budget_usd = Column(Float, default=1000.0)
    daily_spent_usd = Column(Float, default=0.0)
    monthly_spent_usd = Column(Float, default=0.0)
    last_reset_day = Column(String(10), nullable=True)   # YYYY-MM-DD
    last_reset_month = Column(String(7), nullable=True)  # YYYY-MM

    # Rate limits (None = global defaults)
    rpm_limit = Column(Integer, nullable=True)
    tpm_limit = Column(Integer, nullable=True)

    created_at = Column(DateTime, server_default=func.now())


class User(Base):
    """
    User entity within a Team.
    Users have individual spend tracking and roles.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(200), nullable=True, index=True)
    team_id = Column(String(100), nullable=True, index=True)  # FK to teams.team_id
    role = Column(String(50), default="developer", index=True)  # admin | developer | read_only

    # Per-user budget
    daily_budget_usd = Column(Float, nullable=True)  # None = no individual limit
    daily_spent_usd = Column(Float, default=0.0)
    last_reset_day = Column(String(10), nullable=True)

    created_at = Column(DateTime, server_default=func.now())


class ModelAlias(Base):
    """
    Maps a friendly alias (e.g. 'gpt-4-production') to a real provider model.
    Optional team-scoping for per-team model mappings.
    """
    __tablename__ = "model_aliases"

    id = Column(Integer, primary_key=True, index=True)
    alias = Column(String(200), nullable=False, index=True)
    target_model = Column(String(200), nullable=False)
    provider_override = Column(String(100), nullable=True)  # Force specific provider
    team_id = Column(String(100), nullable=True, index=True)  # None = global alias
    created_at = Column(DateTime, server_default=func.now())


# ── Phase 2: Online Router Learning & Feedback ─────────────────────────────


class RouterTrainingLabel(Base):
    """
    Stores labeled training examples for the AI Router.
    Labels can come from:
      - Human feedback (via POST /api/v1/feedback)
      - Inferred from request quality + model routing decisions
    Used by RouterTrainer to retrain the complexity classifier.
    """
    __tablename__ = "router_training_labels"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    request_log_id = Column(Integer, nullable=True, index=True)  # FK to request_logs.id
    prompt_snippet = Column(String(500), nullable=True)          # Prompt text used for feature extraction
    model_requested = Column(String(100), nullable=True)
    complexity_label = Column(String(20), nullable=False, index=True)  # "low" | "medium" | "high"
    label_source = Column(String(50), default="inferred")              # "human" | "inferred" | "feedback"
    confidence = Column(Float, default=1.0)                            # 0.0–1.0 confidence in the label


class RouterTrainingRun(Base):
    """
    Audit trail of each router retraining run.
    Records accuracy before/after swap, sample sizes, and timing.
    Enables debugging and performance tracking over time.
    """
    __tablename__ = "router_training_runs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, server_default=func.now(), index=True)

    samples_trained = Column(Integer, default=0)
    samples_evaluated = Column(Integer, default=0)
    current_accuracy = Column(Float, nullable=True)   # Accuracy of model BEFORE training
    new_accuracy = Column(Float, nullable=True)        # Accuracy of NEW model on held-out set
    model_swapped = Column(Boolean, default=False)     # Whether the new model was deployed
    training_time_ms = Column(Integer, default=0)


class RequestFeedback(Base):
    """
    Per-request quality feedback from end users.
    Structured ratings feed directly into the router training loop:
      - Negative ratings on cheap models → label as higher complexity
      - Negative ratings on expensive models → flag for prompt engineering
    """
    __tablename__ = "request_feedback"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    request_log_id = Column(Integer, nullable=False, index=True)  # FK to request_logs.id
    request_id = Column(String(100), nullable=True, index=True)    # The chatcmpl-xxx id

    # Rating: 1 = good, -1 = bad, 0 = neutral
    rating = Column(Integer, nullable=False)

    # Structured issue category (optional)
    issue = Column(String(50), nullable=True)  # wrong|slow|incomplete|hallucinated|off-topic

    # Free text note (optional)
    note = Column(Text, nullable=True)

    # Attribution
    user_id = Column(String(100), nullable=True, index=True)
    team_id = Column(String(100), nullable=True, index=True)
