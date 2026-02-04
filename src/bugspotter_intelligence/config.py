from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # === Database ===
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "bugspotter_intelligence"
    database_user: str = "postgres"
    database_password: str = "postgres"

    # === Authentication ===
    auth_enabled: bool = Field(
        default=True,
        description="Enable API key authentication (set to false for initial setup)"
    )
    api_key_prefix: str = Field(
        default="bsi_",
        description="Prefix for generated API keys"
    )

    # === Redis (Rate Limiting) ===
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str | None = None
    redis_db: int = 0
    redis_ssl: bool = False

    # === Rate Limiting ===
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_default_rpm: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Default requests per minute"
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limit window in seconds"
    )

    # === CORS ===
    cors_allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins. Use specific domains in production, never '*'"
    )

    # === LLM Providers ===
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: float = 120.0
    anthropic_api_key: str | None = None
    claude_model: str = "claude-sonnet-4-20250514"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4"
    log_level: str = "INFO"
    debug: bool = False
    embedding_provider: str = "local"  # local, openai
    embedding_model: str | None = None  # Provider-specific model name

    # === Caching ===
    cache_enabled: bool = Field(
        default=True,
        description="Enable Redis caching for search results and embeddings"
    )
    cache_ttl_fast_search: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="TTL for fast search cache in seconds"
    )
    cache_ttl_smart_search: int = Field(
        default=900,
        ge=1,
        le=7200,
        description="TTL for smart search cache in seconds"
    )
    cache_ttl_embedding: int = Field(
        default=3600,
        ge=1,
        le=86400,
        description="TTL for embedding cache in seconds"
    )

    # === Smart Search ===
    smart_search_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout in seconds for LLM reranking in smart search"
    )
    smart_search_candidate_limit: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of candidates to retrieve for LLM reranking"
    )

    #=== Similarity and Deduplication Settings ===
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for finding similar bugs (0.0-1.0)"
    )

    duplicate_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for marking as duplicate (0.0-1.0)"
    )

    max_similar_bugs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of similar bugs to return"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False  # DATABASE_URL = database_url
    )

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"

    @property
    def redis_url(self) -> str:
        """Build Redis connection URL"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        protocol = "rediss" if self.redis_ssl else "redis"
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
