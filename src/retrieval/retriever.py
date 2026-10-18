"""Glue between the paper store, the embedder and the vector store."""
from __future__ import annotations

import logging
from typing import List

from src.ingestion.paper_store import PaperStore
from src.models import Paper, RetrievedPaper
from src.retrieval.embeddings import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class PaperRetriever:
    def __init__(
        self,
        paper_store: PaperStore,
        embedder: Embedder,
        vector_store: VectorStore,
        min_score: float = 0.0,
    ) -> None:
        self.paper_store = paper_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.min_score = min_score

    def index_papers(self, papers: List[Paper]) -> int:
        """Embed and index papers (paper metadata must already be stored)."""
        if not papers:
            return 0
        vectors = self.embedder.embed([p.text_for_embedding for p in papers])
        self.vector_store.add([p.uid for p in papers], vectors)
        logger.info("Indexed %d paper(s); index size now %d", len(papers), self.vector_store.count())
        return len(papers)

    def reindex_all(self) -> int:
        self.vector_store.clear()
        return self.index_papers(self.paper_store.all())

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPaper]:
        """Top-k papers whose score is strictly above ``min_score``."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        query_vec = self.embedder.embed([query])[0]
        results: List[RetrievedPaper] = []
        for uid, score in self.vector_store.search(query_vec, k):
            if score <= self.min_score:
                continue
            paper = self.paper_store.get(uid)
            if paper is None:
                logger.warning("Vector index references unknown paper %s (run reindex)", uid)
                continue
            results.append(RetrievedPaper(paper=paper, score=score))
        return results
