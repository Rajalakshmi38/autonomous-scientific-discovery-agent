"""Shared data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Paper:
    """One research paper record, source-agnostic (arXiv now, PubMed later)."""

    source: str                      # "arxiv" | "pubmed"
    paper_id: str                    # id without version, e.g. "2401.01234"
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    published: str = ""              # ISO timestamp as given by the source
    updated: str = ""
    url: str = ""                    # human-readable landing page
    pdf_url: str = ""
    categories: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    version: Optional[int] = None

    @property
    def uid(self) -> str:
        """Globally unique, citable identifier, e.g. ``arxiv:2401.01234``."""
        return f"{self.source}:{self.paper_id}"

    @property
    def year(self) -> str:
        return self.published[:4] if self.published else ""

    @property
    def text_for_embedding(self) -> str:
        return f"{self.title}. {self.abstract}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        return cls(**data)


@dataclass
class RetrievedPaper:
    paper: Paper
    score: float
    captions: List[str] = field(default_factory=list)   # figure captions, when the paper's figures were extracted
