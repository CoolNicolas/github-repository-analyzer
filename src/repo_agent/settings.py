"""Application settings and bounded GitHub retrieval defaults."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    github_request_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    github_token: SecretStr | None = None
    github_max_files: int = Field(default=20, gt=0, le=100)
    github_max_file_bytes: int = Field(default=65_536, gt=0, le=1_048_576)
    github_max_total_bytes: int = Field(default=524_288, gt=0, le=5_242_880)
    llm_enabled: bool = False
    llm_provider: Literal["mock", "openai_compatible"] = "mock"
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_request_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    llm_max_context_tokens: int = Field(default=12_000, ge=256, le=64_000)
    llm_max_output_tokens: int = Field(default=1_000, ge=32, le=8_192)
    repository_cache_ttl_seconds: int = Field(default=600, ge=0, le=3_600)
    rag_sqlite_path: str = "data/repository_index.sqlite3"
    rag_max_files: int = Field(default=40, gt=20, le=500)
    rag_max_file_bytes: int = Field(default=131_072, gt=0, le=1_048_576)
    rag_max_total_bytes: int = Field(default=5_242_880, gt=0, le=52_428_800)
    rag_max_requests: int = Field(default=50, ge=5, le=500)
    rag_max_chunks: int = Field(default=1_000, gt=0, le=20_000)
    rag_chunk_lines: int = Field(default=80, ge=10, le=500)
    rag_chunk_overlap_lines: int = Field(default=15, ge=0, le=100)
    rag_retrieved_chunks: int = Field(default=10, gt=0, le=100)
    rag_diagnostics_enabled: bool = False
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
