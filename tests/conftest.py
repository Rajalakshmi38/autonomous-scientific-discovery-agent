"""Shared test setup: make every test independent of the developer's personal configuration."""
import pytest

import config.settings as settings_module

# Every environment variable that config/settings.py reads.
_SETTING_VARS = [
    "LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_BASE_URL", "LLM_TEMPERATURE", "OLLAMA_NUM_CTX",
    "LLM_MAX_TOKENS", "LLM_TIMEOUT_SECONDS", "EMBEDDING_BACKEND", "EMBEDDING_MODEL",
    "OLLAMA_EMBEDDING_MODEL", "RETRIEVAL_MIN_SCORE", "ARXIV_CONTACT", "ARXIV_TIMEOUT_SECONDS",
    "ARXIV_RETRIES", "ARXIV_DELAY_SECONDS", "STORAGE_DIR", "DATA_DIR", "LOG_LEVEL",
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