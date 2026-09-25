from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "OptiLLM"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"  # text | json

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

    # Extended Providers
    GROQ_API_KEY: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""  # e.g. https://myresource.openai.azure.com
    AZURE_OPENAI_API_VERSION: str = "2024-02-01"
    MISTRAL_API_KEY: str = ""
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"

    # Storage paths
    FAISS_INDEX_PATH: str = "faiss_store/index.faiss"

    # Security & Gateway Controls
    API_KEY_AUTH_ENABLED: bool = False
    OPTILLM_API_KEYS: str = "sk-optillm-dev-key"  # Comma-separated allowed keys
    RATE_LIMIT_PER_MINUTE: int = 60
    TPM_LIMIT_PER_KEY: int = 100000  # Tokens per minute global default per key
    CORS_ORIGINS: str = "*"  # Comma-separated list of allowed origins or "*"
    ALLOWED_PROVIDERS: str = "openai,gemini,anthropic,ollama,groq,azure,mistral,bedrock"
    ALLOWED_MODELS: str = "*"

    # Guardrails & PII
    GUARDRAILS_ENABLED: bool = True
    GUARDRAIL_PII_ENABLED: bool = True
    GUARDRAIL_CONTENT_SAFETY_ENABLED: bool = False
    GUARDRAIL_PROMPT_INJECTION_ENABLED: bool = False
    PRESIDIO_ENABLED: bool = True
    PRESIDIO_MODEL: str = "en_core_web_sm"

    # Secrets Management Backend (Phase 5.3)
    SECRETS_BACKEND: str = "env"  # env | file | aws | vault
    SECRETS_DIR: str = "/run/secrets"
    AWS_SECRET_NAME: str = "optillm/config"
    VAULT_ADDR: str = ""
    VAULT_TOKEN: str = ""
    VAULT_SECRET_PATH: str = "secret/data/optillm"

    # Global Rate Limiting & Protection (Phase 5.4)
    GLOBAL_RATE_LIMIT_PER_MINUTE: int = 1200  # Gateway-wide RPM limit across all clients
    RATE_LIMIT_BURST_FACTOR: float = 1.5      # Multiplier for token bucket burst capacity
    AUTO_BLOCKLIST_ENABLED: bool = True       # Block repeated 429 abusers
    AUTO_BLOCKLIST_THRESHOLD: int = 5         # Violations within 60s before auto-block
    AUTO_BLOCKLIST_DURATION_SECONDS: int = 300 # Duration in seconds for client block
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

    # AI Router & Intelligent Routing
    ROUTING_MODE: str = "shadow"  # rule_based | ai | shadow
    AI_ROUTER_CONFIDENCE_THRESHOLD: float = 0.70
    AI_ROUTER_MODEL_PATH: str = "ai_router.joblib"

    # Online Router Learning Loop
    ROUTER_AUTO_RETRAIN: bool = True              # Enable automatic background retraining
    ROUTER_RETRAIN_INTERVAL_REQUESTS: int = 1000  # Retrain after this many new requests
    ROUTER_MIN_TRAINING_SAMPLES: int = 30         # Minimum samples needed to train

    # RAG Support (Phase 12)
    QDRANT_URL: str = ":memory:"
    RAG_DEFAULT_CHUNK_SIZE: int = 500
    RAG_DEFAULT_CHUNK_OVERLAP: int = 50

    # Observability Callbacks
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    SLACK_WEBHOOK_URL: str = ""
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""  # e.g. http://localhost:4318

    # Config YAML
    CONFIG_YAML_PATH: str = "config.yaml"
    CONFIG_HOT_RELOAD: bool = True

    # Evaluation Engine
    EVAL_ENABLED: bool = True                # Run heuristic evaluation on every request
    EVAL_LLM_ENABLED: bool = False           # Enable LLM-as-judge (costs ~$0.0001/eval)
    EVAL_SAMPLE_RATE: float = 0.05           # Fraction of requests scored by LLM judge (0.0–1.0)
    EVAL_LLM_MODEL: str = "gpt-4o-mini"     # Cheap-but-capable model to use as judge

    # Input Limits & Protection (Phase 3.5)
    MAX_REQUEST_BYTES: int = 1_048_576       # 1 MB max raw request body size
    MAX_MESSAGES_PER_REQUEST: int = 200      # Max messages in a single chat request
    MAX_PROMPT_LENGTH: int = 500_000         # Max characters per prompt / message

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")




settings = Settings()
