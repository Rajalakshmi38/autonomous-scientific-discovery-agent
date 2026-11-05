"""LLM provider registry. To add a provider later, write one ``build`` function,
register it below, and add its name to VALID_LLM_PROVIDERS in config/settings.py.
Nothing else in the project needs to change.

Free options only are provided out of the box:
  * ``ollama`` - free, open-source local LLM server (no account, no API key)
  * ``test``   - rule-based MOCK for plumbing tests; it is NOT an LLM
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import ConfigError, Settings
from src.llm.stub_model import RuleBasedStubChatModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    build: Callable[[Settings], BaseChatModel]
    describe: Callable[[Settings], str]
    is_mock: bool = False
    needs_local_server: bool = False


def _build_test(settings: Settings) -> BaseChatModel:
    logger.warning("LLM_PROVIDER=test: using the rule-based MOCK. No genuine LLM reasoning.")
    return RuleBasedStubChatModel()


def _build_ollama(settings: Settings) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ConfigError(
            "LLM_PROVIDER=ollama needs the Python package: pip install -r requirements-llm.txt "
            "(and the Ollama app installed and running; see the README)"
        ) from exc
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.llm_max_tokens,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )


_PROVIDERS: Dict[str, ProviderSpec] = {}


def register_provider(spec: ProviderSpec) -> None:
    _PROVIDERS[spec.name] = spec


register_provider(
    ProviderSpec("test", _build_test, lambda s: "TEST MODE (rule-based mock, no LLM reasoning)", is_mock=True)
)
register_provider(
    ProviderSpec("ollama", _build_ollama, lambda s: f"ollama:{s.ollama_model} (local, free)", needs_local_server=True)
)


def available_providers() -> List[str]:
    return sorted(_PROVIDERS)


def get_provider_spec(name: str) -> ProviderSpec:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ConfigError(f"Unsupported LLM_PROVIDER {name!r}. Available: {available_providers()}") from None


def get_llm(settings: Settings) -> BaseChatModel:
    return get_provider_spec(settings.llm_provider).build(settings)


def describe_llm(settings: Settings) -> str:
    """Human-readable label stored with every result so provenance is clear."""
    return get_provider_spec(settings.llm_provider).describe(settings)
