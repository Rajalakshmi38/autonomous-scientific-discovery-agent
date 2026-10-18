"""Central configuration.

All values come from environment variables (optionally loaded from a local
``.env`` file). No secrets live in source code. Nothing here is required for
the default ``test`` LLM provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keep in sync with the registry in src/llm/factory.py (a unit test enforces this).
VALID_LLM_PROVIDERS = {"test", "ollama"}
VALID_EMBEDDING_BACKENDS = {"hashing", "ollama", "sentence-transformers"}


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "test"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.2
    ollama_num_ctx: int = 4096          # context window; Ollama's own default can truncate long prompts
    llm_max_tokens: int = 700           # cap on generated tokens per call
    llm_timeout_seconds: float = 300.0  # local CPU inference can be slow; never hang forever

    embedding_backend: str = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # sentence-transformers backend
    ollama_embedding_model: str = "nomic-embed-text"                  # ollama backend
    retrieval_min_score: float = 0.0

    arxiv_contact: str = ""
    arxiv_timeout_seconds: float = 30.0
    arxiv_retries: int = 3
    arxiv_delay_seconds: float = 3.0

    storage_dir: Path = PROJECT_ROOT / "storage"
    data_dir: Path = PROJECT_ROOT / "data"
    log_level: str = "INFO"

    # ---- derived paths -------------------------------------------------
    @property
    def papers_db_path(self) -> Path:
        return self.storage_dir / "papers.db"

    @property
    def vector_dir(self) -> Path:
        return self.storage_dir / "vectors"

    @property
    def runs_dir(self) -> Path:
        return self.storage_dir / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.storage_dir / "logs"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def is_test_mode(self) -> bool:
        return self.llm_provider == "test"


def _resolve_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _num(env: Mapping[str, str], key: str, default, cast):
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a {cast.__name__}, got {raw!r}") from exc


def load_settings(
    env: Optional[Mapping[str, str]] = None, load_env_file: bool = True
) -> Settings:
    """Build Settings from the environment (and ``.env`` if present)."""
    if env is None:
        if load_env_file:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ

    provider = env.get("LLM_PROVIDER", "test").strip().lower()
    if provider not in VALID_LLM_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER={provider!r} is not supported. Choose from {sorted(VALID_LLM_PROVIDERS)}."
        )
    backend = env.get("EMBEDDING_BACKEND", "hashing").strip().lower()
    if backend not in VALID_EMBEDDING_BACKENDS:
        raise ConfigError(
            f"EMBEDDING_BACKEND={backend!r} is not supported. Choose from {sorted(VALID_EMBEDDING_BACKENDS)}."
        )

    retries = _num(env, "ARXIV_RETRIES", 3, int)
    if retries < 1:
        raise ConfigError("ARXIV_RETRIES must be >= 1")

    return Settings(
        llm_provider=provider,
        ollama_model=env.get("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_base_url=env.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_temperature=_num(env, "LLM_TEMPERATURE", 0.2, float),
        ollama_num_ctx=_num(env, "OLLAMA_NUM_CTX", 4096, int),
        llm_max_tokens=_num(env, "LLM_MAX_TOKENS", 700, int),
        llm_timeout_seconds=_num(env, "LLM_TIMEOUT_SECONDS", 300.0, float),
        embedding_backend=backend,
        embedding_model=env.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        ollama_embedding_model=env.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        retrieval_min_score=_num(env, "RETRIEVAL_MIN_SCORE", 0.0, float),
        arxiv_contact=env.get("ARXIV_CONTACT", "").strip(),
        arxiv_timeout_seconds=_num(env, "ARXIV_TIMEOUT_SECONDS", 30.0, float),
        arxiv_retries=retries,
        arxiv_delay_seconds=_num(env, "ARXIV_DELAY_SECONDS", 3.0, float),
        storage_dir=_resolve_dir(env.get("STORAGE_DIR", "storage")),
        data_dir=_resolve_dir(env.get("DATA_DIR", "data")),
        log_level=env.get("LOG_LEVEL", "INFO").upper(),
    )
