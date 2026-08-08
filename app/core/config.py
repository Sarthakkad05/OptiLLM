from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "OptiLLM"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Database — defaults to local SQLite for zero-config local dev
    DATABASE_URL: str = "sqlite:///./optillm.db"

    # LLM Providers
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4o"
    REQUEST_TIMEOUT_SECONDS: float = 60.0

    # Storage paths
    FAISS_INDEX_PATH: str = "faiss_store/index.faiss"

    # Security & Gateway Controls
    API_KEY_AUTH_ENABLED: bool = False
    OPTILLM_API_KEYS: str = "sk-optillm-dev-key"  # Comma-separated allowed keys
    RATE_LIMIT_PER_MINUTE: int = 60
    CORS_ORIGINS: str = "*"  # Comma-separated list of allowed origins or "*"
    ALLOWED_PROVIDERS: str = "openai,gemini,anthropic,ollama"
    ALLOWED_MODELS: str = "*"
    # High Availability & Routing Strategies
    ROUTING_STRATEGY: str = (
        "cost_optimized"  # cost_optimized | round_robin | least_latency
    )
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 3
    CIRCUIT_BREAKER_RECOVERY_TIME: float = 30.0
    # Distributed Cache Settings
    REDIS_URL: str = ""  # e.g., redis://localhost:6379/0
    CACHE_TTL_SECONDS: int = 86400  # 24 hours
    CACHE_NAMESPACE: str = "default"
    CACHE_SIMILARITY_THRESHOLD: float = 0.90

    # AI Router & Intelligent Routing (Phase 10)
    ROUTING_MODE: str = "shadow"  # rule_based | ai | shadow
    AI_ROUTER_CONFIDENCE_THRESHOLD: float = 0.70
    AI_ROUTER_MODEL_PATH: str = "ai_router.joblib"

    # RAG Support (Phase 12)
    QDRANT_URL: str = ":memory:"
    RAG_DEFAULT_CHUNK_SIZE: int = 500
    RAG_DEFAULT_CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")




settings = Settings()
