"""Common interface for paper sources so PubMed (or others) can be added later."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.models import Paper


class IngestionError(RuntimeError):
    """Raised when a paper source cannot be reached or returns unusable data."""


class PaperSource(ABC):
    name: str = "unknown"

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Return papers matching ``query``. Empty list means no results."""
