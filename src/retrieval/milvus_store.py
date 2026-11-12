"""Milvus adapter for the vector store (the plan names Milvus or Pinecone; Milvus is open source and free).

``uri`` is a Milvus server, for example ``http://localhost:19530`` (run Milvus with Docker), or a local file such as
``storage/milvus.db``, which uses Milvus Lite. Milvus Lite exists for Linux and macOS only, so on Windows use a server.

One collection is used per embedder, so a new embedder starts with an empty index. Run ``reindex`` after switching.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.retrieval.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)
_BATCH = 500


def collection_name(embedder_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", embedder_name or "default")
    return f"asda_papers_{safe}"[:200]


class MilvusVectorStore(VectorStore):
    def __init__(self, uri: str, embedder_name: str, client: Optional[Any] = None) -> None:
        self.uri = uri
        self.collection = collection_name(embedder_name)
        self.client = client if client is not None else self._connect(uri)

    @staticmethod
    def _connect(uri: str) -> Any:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise VectorStoreError("Milvus needs the Python package: pip install -r requirements-milvus.txt") from exc
        try:
            return MilvusClient(uri)
        except Exception as exc:     # pymilvus raises several different error types for connection problems
            hint = (" Milvus Lite (a local file) is not available on Windows. Start a Milvus server with Docker and use "
                    "http://localhost:19530") if not uri.startswith("http") else " Is the Milvus server running?"
            raise VectorStoreError(f"Could not open Milvus at {uri!r}: {exc}.{hint}") from exc

    def _call(self, what: str, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Milvus {what} failed: {exc}") from exc

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
            raise VectorStoreError("ids and vectors must have matching lengths (vectors must be 2-D)")
        if len(ids) == 0:
            return
        if not self._call("check", self.client.has_collection, self.collection):
            self._call("create", self.client.create_collection, self.collection, dimension=int(vectors.shape[1]),
                       metric_type="COSINE", id_type="string", max_length=256, auto_id=False)
        rows = [{"id": str(i), "vector": v.tolist()} for i, v in zip(ids, vectors)]
        for start in range(0, len(rows), _BATCH):
            self._call("insert", self.client.upsert, self.collection, rows[start:start + _BATCH])

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        if k <= 0 or not self._call("check", self.client.has_collection, self.collection):
            return []
        hits = self._call("search", self.client.search, self.collection,
                          data=[np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()], limit=k, output_fields=[])
        return [(str(h["id"]), float(h["distance"])) for h in (hits[0] if hits else [])]

    def count(self) -> int:
        if not self._call("check", self.client.has_collection, self.collection):
            return 0
        rows = self._call("count", self.client.query, self.collection, filter="", output_fields=["count(*)"])
        return int(rows[0]["count(*)"]) if rows else 0

    def clear(self) -> None:
        if self._call("check", self.client.has_collection, self.collection):
            self._call("drop", self.client.drop_collection, self.collection)
