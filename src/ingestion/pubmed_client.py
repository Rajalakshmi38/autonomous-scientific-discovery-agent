"""PubMed placeholder (NOT IMPLEMENTED).

Exists so the rest of the code can depend on the ``PaperSource`` interface.
A future implementation would call NCBI E-utilities (esearch + efetch) and
return ``Paper(source="pubmed", ...)`` records. Until then this class raises
``NotImplementedError`` rather than pretending to work.
"""
from __future__ import annotations

from typing import List

from src.ingestion.base import PaperSource
from src.models import Paper


class PubMedClient(PaperSource):
    name = "pubmed"

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        raise NotImplementedError("PubMed ingestion is planned but not implemented yet.")
