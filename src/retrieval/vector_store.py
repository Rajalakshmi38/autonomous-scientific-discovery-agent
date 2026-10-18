"""Vector store interface + a small NumPy development adapter.

``VectorStore`` is the seam where free, self-hosted adapters (Chroma, FAISS, Milvus) plug in
later: implement the same four methods and nothing else changes.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    pass


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def clear(self) -> None: ...


class NumpyVectorStore(VectorStore):
    """Brute-force cosine search; fine for thousands of papers, not millions.

    Vectors are assumed L2-normalised (our embedders guarantee it), so a dot
    product equals cosine similarity. Persists to ``<dir>/vectors.npy`` and
    ``<dir>/meta.json`` when a directory is given.
    """

    def __init__(
        self, directory: Optional[Path] = None, embedder_name: str = "", load_existing: bool = True
    ) -> None:
        self.directory = Path(directory) if directory else None
        self.embedder_name = embedder_name
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        if self.directory is not None and load_existing:
            self._load()

    # ---- persistence ----------------------------------------------------
    def _load(self) -> None:
        meta_path, vec_path = self.directory / "meta.json", self.directory / "vectors.npy"
        if not (meta_path.exists() and vec_path.exists()):
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != self.embedder_name:
            raise VectorStoreError(
                f"Index was built with embedder {meta.get('embedder')!r} but current embedder is "
                f"{self.embedder_name!r}. Rebuild it with: python -m src.main reindex"
            )
        self._ids = meta["ids"]
        self._matrix = np.load(vec_path)

    def _save(self) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "vectors.npy", self._matrix if self._matrix is not None else np.zeros((0, 0)))
        (self.directory / "meta.json").write_text(
            json.dumps({"embedder": self.embedder_name, "ids": self._ids}), encoding="utf-8"
        )

    # ---- API ---------------------------------------------------------------
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
            raise VectorStoreError("ids and vectors must have matching lengths (vectors must be 2-D)")
        if len(ids) == 0:
            return
        if self._matrix is not None and self._matrix.shape[1] != vectors.shape[1]:
            raise VectorStoreError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self._matrix.shape[1]}"
            )
        # upsert: drop existing rows whose id is being re-added
        incoming = set(ids)
        if self._matrix is not None and incoming & set(self._ids):
            keep = [i for i, existing in enumerate(self._ids) if existing not in incoming]
            self._ids = [self._ids[i] for i in keep]
            self._matrix = self._matrix[keep]
        self._ids.extend(ids)
        self._matrix = vectors if self._matrix is None or len(self._matrix) == 0 else np.vstack([self._matrix, vectors])
        self._save()

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if self._matrix is None or len(self._ids) == 0 or k <= 0:
            return []
        scores = self._matrix @ np.asarray(query_vector, dtype=np.float32).reshape(-1)
        top = np.argsort(-scores)[:k]
        return [(self._ids[i], float(scores[i])) for i in top]

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids, self._matrix = [], None
        self._save()
