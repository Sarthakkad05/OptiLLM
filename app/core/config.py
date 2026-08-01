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
    DEFAULT_PROVIDER: str = "openai"
    DEFAULT_MODEL: str = "gpt-4o"

    # Storage paths
    FAISS_INDEX_PATH: str = "faiss_store/index.faiss"

    # Security & Gateway Controls
    API_KEY_AUTH_ENABLED: bool = False
    OPTILLM_API_KEYS: str = "sk-optillm-dev-key"  # Comma-separated allowed keys
    RATE_LIMIT_PER_MINUTE: int = 60

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
