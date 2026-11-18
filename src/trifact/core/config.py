"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIFACT_", env_file=".env", extra="ignore")

    postgres_dsn: str = "postgresql+psycopg://trifact:trifact@localhost:5432/trifact"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_bucket_raw: str = "trifact-raw"

    inference_url: str = "http://localhost:8001"
    llm_url: str = "http://localhost:8080/v1"
    llm_model: str = "trifact-llm"

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_alias: str = "champion"

    # retrieval
    candidates_k: int = 50
    rrf_k: int = 60
    context_k: int = 8

    # verification
    nei_ratio_threshold: float = 0.3


settings = Settings()
