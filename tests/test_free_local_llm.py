"""Tests for the free/local LLM + embedding paths.

IMPORTANT HONESTY NOTE: real Ollama is not available in the test environment. These tests run
against a tiny FAKE server that imitates Ollama's documented HTTP API. They prove that our code
sends sensible requests (settings, prompts) and handles responses/failures. They do NOT prove
that a real model gives good answers, and the fake's embeddings are not semantic.
"""
import json
import pathlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from config.settings import VALID_LLM_PROVIDERS, load_settings
from src.agents.base import AgentError
from src.agents.literature_reviewer import LiteratureReviewerAgent
from src.llm.factory import available_providers, describe_llm, get_llm, get_provider_spec
from src.llm.ollama_check import check_ollama
from src.main import main
from src.models import Paper, RetrievedPaper
from src.retrieval.embeddings import EmbeddingError, OllamaEmbedder, get_embedder
from src.utils.system_info import SystemInfo, get_system_info, parse_nvidia_smi, recommend

pytest.importorskip("langchain_ollama", reason="pip install -r requirements-llm.txt")


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):            # keep test output quiet
        pass

    def _send(self, status, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "llama3.2:3b"}, {"name": "nomic-embed-text:latest"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, req))
        if self.path == "/api/chat":
            base = {"model": req["model"], "created_at": "2024-01-01T00:00:00Z"}
            lines = [
                {**base, "message": {"role": "assistant", "content": "[FAKE-SERVER] hello "}, "done": False},
                {**base, "message": {"role": "assistant", "content": "world"}, "done": True, "done_reason": "stop"},
            ]
            self._send(200, ("\n".join(json.dumps(l) for l in lines) + "\n").encode(), "application/x-ndjson")
        elif self.path == "/api/generate":
            self._send(200, {"model": req["model"], "done": True, "done_reason": "unload"})
        elif self.path == "/api/embed":
            if req["model"] == "missing-model":
                self._send(404, {"error": "model not found"})
                return
            vecs = [[float(len(t) % 7 + 1), 1.0, float(i + 1), 0.5] for i, t in enumerate(req["input"])]
            self._send(200, {"model": req["model"], "embeddings": vecs})
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture
def fake_ollama(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()
    server.server_close()


DEAD_URL = "http://127.0.0.1:1"   # nothing listens here


def ollama_settings(url, **extra):
    env = {"LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": url, "OLLAMA_MODEL": "llama3.2:3b", **extra}
    return load_settings(env=env, load_env_file=False)


EVIDENCE = [RetrievedPaper(Paper("arxiv", "1", "Graph networks", "Abstract.", url="u"), 0.5)]


# ------------------------------------------------------------- preflight
def test_check_ollama_model_present_and_missing(fake_ollama):
    assert check_ollama(fake_ollama.url, "llama3.2:3b").ok
    assert check_ollama(fake_ollama.url, "nomic-embed-text").ok            # ':latest' normalisation
    missing = check_ollama(fake_ollama.url, "llama3.1:8b")
    assert missing.reachable and not missing.ok and "ollama pull llama3.1:8b" in missing.message


def test_check_ollama_unreachable_gives_install_hint():
    status = check_ollama(DEAD_URL)
    assert not status.reachable and "ollama.com/download" in status.message


# ------------------------------------------------------------- chat model via real langchain-ollama
def test_ollama_provider_sends_expected_request(fake_ollama):
    s = ollama_settings(fake_ollama.url, OLLAMA_NUM_CTX="3072", LLM_MAX_TOKENS="123")
    review = LiteratureReviewerAgent(get_llm(s)).run("graphs", EVIDENCE)
    assert review.summary.startswith("[FAKE-SERVER] hello world")
    path, req = next(r for r in fake_ollama.requests if r[0] == "/api/chat")
    assert req["model"] == "llama3.2:3b"
    assert req["options"]["num_ctx"] == 3072 and req["options"]["num_predict"] == 123
    assert req["messages"][0]["role"] == "system" and "untrusted" in req["messages"][0]["content"]
    assert "[arxiv:1]" in req["messages"][1]["content"]


def test_ollama_down_becomes_clear_agent_error():
    llm = get_llm(ollama_settings(DEAD_URL, LLM_TIMEOUT_SECONDS="2"))
    with pytest.raises(AgentError, match="model call failed"):
        LiteratureReviewerAgent(llm).run("graphs", EVIDENCE)


def test_describe_llm_labels_local_free_provider():
    assert "local, free" in describe_llm(ollama_settings("http://x"))


# ------------------------------------------------------------- registry / modularity
def test_registry_matches_settings_constant():
    assert set(available_providers()) == VALID_LLM_PROVIDERS
    assert get_provider_spec("test").is_mock and not get_provider_spec("ollama").is_mock


def test_no_paid_api_keys_referenced_in_source():
    root = pathlib.Path(__file__).resolve().parent.parent
    pattern = re.compile(r"api[_-]?key|openai|anthropic", re.IGNORECASE)
    hits = [
        str(p) for folder in ("src", "config") for p in (root / folder).rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert hits == [], f"Unexpected API-key/paid-provider references: {hits}"


# ------------------------------------------------------------- Ollama embeddings
def test_ollama_embedder_shape_norm_and_batching(fake_ollama):
    emb = OllamaEmbedder("nomic-embed-text", fake_ollama.url)
    out = emb.embed([f"text {i}" for i in range(20)])         # 20 > batch size 16 -> 2 requests
    assert out.shape == (20, 4) and emb.dim == 4
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert len([r for r in fake_ollama.requests if r[0] == "/api/embed"]) == 2


def test_ollama_embedder_errors_are_explicit(fake_ollama):
    with pytest.raises(EmbeddingError, match="ollama pull missing-model"):
        OllamaEmbedder("missing-model", fake_ollama.url).embed(["x"])
    with pytest.raises(EmbeddingError, match="Cannot reach Ollama"):
        OllamaEmbedder("m", DEAD_URL, timeout=2).embed(["x"])


def test_embedder_factory_selects_ollama():
    s = load_settings(env={"EMBEDDING_BACKEND": "ollama"}, load_env_file=False)
    assert get_embedder(s).name == "ollama-nomic-embed-text"


# ------------------------------------------------------------- hardware guidance
def test_parse_nvidia_smi():
    assert parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288\n") == ("NVIDIA GeForce RTX 3060", 12.0)
    assert parse_nvidia_smi("garbage") is None and parse_nvidia_smi("") is None


@pytest.mark.parametrize(
    "ram,vram,tier",
    [(4, None, "mock-only"), (6.5, None, "mock-only"), (7.3, None, "small"), (8, None, "small"), (15.4, None, "medium"), (16, None, "medium"), (32, 8, "large"), (None, None, "unknown")],
)
def test_recommendation_tiers(ram, vram, tier):
    assert recommend(SystemInfo("x", 4, ram, "gpu" if vram else None, vram)).tier == tier


def test_get_system_info_does_not_crash():
    assert get_system_info().os_name


# ------------------------------------------------------------- CLI
def test_cli_run_ollama_unreachable_fails_fast_with_hint(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("OLLAMA_BASE_URL", DEAD_URL)
    assert main(["run", "--topic", "x", "--provider", "ollama"]) == 1
    err = capsys.readouterr().err
    assert "Cannot reach Ollama" in err and "--provider test" in err


def test_cli_check_llm_and_check_system(monkeypatch, tmp_path, fake_ollama, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", fake_ollama.url)
    assert main(["check-llm"]) == 0
    assert "available" in capsys.readouterr().out
    assert main(["check-system"]) == 0
    assert "RAM" in capsys.readouterr().out


# ------------------------------------------------------------- freeing memory before the sandbox starts
def test_unload_model_sends_keep_alive_zero(fake_ollama):
    from src.llm.ollama_check import unload_model
    assert unload_model(fake_ollama.url, "llama3.2:3b") is True
    path, req = fake_ollama.requests[-1]
    assert path == "/api/generate" and req == {"model": "llama3.2:3b", "keep_alive": 0}


def test_unload_model_reports_failure_instead_of_raising():
    from src.llm.ollama_check import unload_model
    assert unload_model(DEAD_URL, "llama3.2:3b", timeout=2) is False
