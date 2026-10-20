"""Shared test setup: make every test independent of the developer's personal configuration."""
import logging

import pytest

import config.settings as settings_module

# Every environment variable that config/settings.py reads.
_SETTING_VARS = [
    "LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "LLM_TEMPERATURE", "OLLAMA_NUM_CTX",
    "LLM_MAX_TOKENS", "LLM_TIMEOUT_SECONDS", "EMBEDDING_BACKEND", "EMBEDDING_MODEL",
    "OLLAMA_EMBEDDING_MODEL", "RETRIEVAL_MIN_SCORE", "ARXIV_CONTACT", "ARXIV_TIMEOUT_SECONDS",
    "ARXIV_RETRIES", "ARXIV_DELAY_SECONDS", "STORAGE_DIR", "DATA_DIR", "LOG_LEVEL",
    "CONTEXT_CHAR_BUDGET", "CITATION_RETRIES", "MAX_REPAIR_RETRIES", "DOCKER_COMMAND", "SANDBOX_IMAGE",
    "SANDBOX_TIMEOUT_SECONDS", "SANDBOX_MEMORY_MB", "SANDBOX_CPUS", "SANDBOX_PIDS_LIMIT", "SANDBOX_MAX_OUTPUT_KB",
]


@pytest.fixture(autouse=True)
def isolate_from_local_config(monkeypatch):
    """Tests must not read the real .env file or inherit settings from the shell.

    Without this, a personal .env (for example LLM_PROVIDER=ollama) changes test results.
    Tests that need a setting set it themselves with monkeypatch.setenv(...).
    """
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *args, **kwargs: False)
    for name in _SETTING_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def restore_logging():
    """CLI tests call setup_logging(), which replaces the root logger's handlers. Undo that after each test."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    for handler in handlers:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)
