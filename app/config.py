from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AI Agronomy Assistant"
    app_env: str = "local"
    database_url: str = "postgresql+asyncpg://agronomy:agronomy@localhost:5432/agronomy"
    sync_database_url: str = "postgresql+psycopg://agronomy:agronomy@localhost:5432/agronomy"
    embedding_dimension: int = Field(default=1024, ge=1)
    llm_provider: str = "google"
    llm_model: str = "gemini-2.5-flash"
    gemini_api_key: str | None = None
    vision_provider: str = "google"
    embeddings_provider: str = "google"
    embeddings_model: str = "text-embedding-004"
    reranker_provider: str = "stub"


@lru_cache
def get_settings() -> Settings:
    return Settings()
