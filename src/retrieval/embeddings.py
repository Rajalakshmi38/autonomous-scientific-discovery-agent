"""Embedding backends.

* ``HashingEmbedder`` (default): dependency-free LEXICAL baseline. It scores
  keyword/phrase overlap, so it is NOT semantic: "car" and "automobile" are
  unrelated to it. Good enough to test the plumbing and for keyword-heavy queries.
* ``OllamaEmbedder`` (optional, free): semantic embeddings from a small local model served by
  Ollama (default ``nomic-embed-text``). No PyTorch needed; Ollama must be running.
* ``SentenceTransformerEmbedder`` (optional, free): semantic embeddings via PyTorch (large install).
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

import numpy as np
import requests

from config.settings import ConfigError, Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "with we our can using use used based via into than then these those their which also".split()
)


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce vectors."""


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 1024) -> None:
        if dim < 16:
            raise ValueError("dim must be >= 16")
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def _tokens(text: str) -> List[str]:
        words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]   # unigrams + bigrams

    def _vectorise(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token, count in Counter(self._tokens(text)).items():
            digest = hashlib.md5(token.encode("utf-8")).digest()   # deterministic across runs
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vectorise(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Optional. Requires ``pip install sentence-transformers`` (large download)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                "EMBEDDING_BACKEND=sentence-transformers needs: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = f"st-{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OllamaEmbedder(Embedder):
    """Calls a local Ollama server (``POST /api/embed``). Free; no API key."""

    BATCH = 16

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout: float = 120.0, session=None) -> None:
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout
        self._http = session or requests
        self.name = f"ollama-{model}"
        self.dim = 0   # learned from the first response

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = []
        for i in range(0, len(texts), self.BATCH):
            batches.append(self._embed_batch(texts[i : i + self.BATCH]))
        return np.vstack(batches)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        try:
            resp = self._http.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Cannot reach Ollama at {self.base_url} for embeddings. Is it running? ({exc})"
            ) from exc
        if resp.status_code != 200:
            raise EmbeddingError(
                f"Ollama embedding request failed (HTTP {resp.status_code}). "
                f"Is the model pulled? Run: ollama pull {self.model}"
            )
        try:
            vectors = np.asarray(resp.json()["embeddings"], dtype=np.float32)
        except (KeyError, ValueError) as exc:
            raise EmbeddingError("Ollama returned an unexpected embedding response.") from exc
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingError("Ollama returned the wrong number of embeddings.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1.0, norms)
        self.dim = int(vectors.shape[1])
        return vectors


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "ollama":
        return OllamaEmbedder(settings.ollama_embedding_model, settings.ollama_base_url)
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    return HashingEmbedder()
