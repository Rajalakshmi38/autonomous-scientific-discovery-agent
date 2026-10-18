"""Preflight check for a local Ollama server (uses only ``requests``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import requests


@dataclass
class OllamaStatus:
    reachable: bool
    models: List[str] = field(default_factory=list)
    model_available: Optional[bool] = None    # None when not checked / unreachable
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.reachable and self.model_available is not False


def _normalise(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def check_ollama(
    base_url: str, model: Optional[str] = None, timeout: float = 3.0,
    session: Optional[requests.Session] = None,
) -> OllamaStatus:
    """Is the Ollama server up, and is ``model`` already pulled?"""
    http = session or requests
    try:
        resp = http.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
    except requests.RequestException:
        return OllamaStatus(
            False,
            message=f"Cannot reach Ollama at {base_url}. Install it from https://ollama.com/download "
                    "and make sure it is running (look for the llama icon in the system tray).",
        )
    if resp.status_code != 200:
        return OllamaStatus(False, message=f"Ollama at {base_url} answered HTTP {resp.status_code}.")
    try:
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except ValueError:
        return OllamaStatus(False, message=f"Ollama at {base_url} returned an unreadable response.")
    if model is None:
        return OllamaStatus(True, names, None, "Ollama is running.")
    available = _normalise(model) in {_normalise(n) for n in names}
    msg = (f"Ollama is running and model '{model}' is available." if available else
           f"Ollama is running but model '{model}' is not downloaded. Run: ollama pull {model}")
    return OllamaStatus(True, names, available, msg)
