"""Central configuration.

All runtime configuration is read from environment variables (optionally via a
`.env` file) and validated by pydantic-settings. Import the shared `settings`
object rather than reading `os.environ` directly, so behaviour stays consistent
across the API, the worker and the CLI.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Env = Literal["local", "dev", "prod"]
SandboxBackend = Literal["subprocess", "docker"]
LLMProvider = Literal["ollama", "openai", "anthropic", "none"]
NetworkPolicy = Literal["deny", "allowlist", "allow"]
DeployTarget = Literal["hf_endpoint", "hf_space", "local_openai"]


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_prefix="SLMFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core ---
    env: Env = "local"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    database_url: str = "sqlite:///./.data/slmforge.db"
    artifact_root: Path = Path("./artifacts")
    upload_root: Path = Path("./uploads")

    # --- Planner LLM (drives the analyzer/planner, not the trained model) ---
    llm_provider: LLMProvider = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "llama3.1:8b"
    llm_api_key: str = ""

    # --- Model sources ---
    ollama_base_url: str = "http://127.0.0.1:11434"
    hf_token: str = ""
    hf_home: Path = Path("./.cache/huggingface")

    # --- Sandbox ---
    sandbox_backend: SandboxBackend = "subprocess"
    sandbox_image: str = "slmforge/sandbox:latest"
    sandbox_cpu_limit: int = 8
    sandbox_mem_limit_mb: int = 16384
    sandbox_timeout_s: int = 7200
    sandbox_network: NetworkPolicy = "deny"

    # --- Training defaults ---
    max_train_minutes: int = 60
    max_vram_gb: int = 8
    default_method: str = "qlora"
    seed: int = 42

    # --- Deployment ---
    hf_push_enabled: bool = False
    hf_org: str = ""
    deploy_target: DeployTarget = "hf_endpoint"
    deploy_require_approval: bool = True

    # --- Web ---
    # `NoDecode` disables pydantic-settings' JSON parsing so a plain
    # comma-separated env value (`http://a,http://b`) is accepted.
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        """Create the runtime directories SLMForge writes to."""
        for path in (self.artifact_root, self.upload_root, self.hf_home):
            Path(path).mkdir(parents=True, exist_ok=True)
        if self.is_sqlite:
            db_path = self.database_url.split("sqlite:///", 1)[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
