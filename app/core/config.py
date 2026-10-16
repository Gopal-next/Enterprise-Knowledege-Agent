"""
Centralized, typed configuration using Pydantic Settings.
All environment variables are validated at startup — if a required
key is missing, the server refuses to start instead of crashing mid-request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Application configuration.
    Values are loaded from .env file and environment variables.
    Environment variables take precedence over .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google / Gemini ──────────────────────────────────────
    google_api_key: str
    gemini_model: str = "gemini-3.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"

    # ── Supabase ─────────────────────────────────────────────
    supabase_url: str
    supabase_key: str

    # ── Qdrant (optional) ────────────────────────────────────
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # ── LangSmith Observability ──────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: Optional[str] = None
    langchain_project: str = "enterprise-knowledge-agent"
    langchain_environment: str = "Production"
    langsmith_disable_run_compression: bool = True

    # ── API Server ───────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    cors_origins: str = "*"

    # ── Security ─────────────────────────────────────────────
    jwt_secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    api_key_header: str = "X-API-Key"
    admin_api_key: str = "eka-admin-key-change-me"

    # ── Rate Limiting ────────────────────────────────────────
    rate_limit_per_minute: int = 30
    rate_limit_burst: int = 5

    # ── Cache ────────────────────────────────────────────────
    cache_ttl_seconds: int = 300
    semantic_cache_threshold: float = 0.92

    # ── Circuit Breaker ──────────────────────────────────────
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60

    # ── Paths ────────────────────────────────────────────────
    pdf_folder: str = str(PROJECT_ROOT / "Data" / "pdfs")
    faiss_index_folder: str = str(PROJECT_ROOT / "Data" / "faiss_index")
    excel_folder: str = str(PROJECT_ROOT / "Data" / "excelfile")
    log_file: str = str(PROJECT_ROOT / "logs" / "app.log")


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Validates all config on first call — fails fast if required keys are missing.
    """
    return Settings()
