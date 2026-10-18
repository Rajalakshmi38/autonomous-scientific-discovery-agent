"""Shared helpers for all agents: evidence formatting, citation checking, error type."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

from src.models import RetrievedPaper

CITATION_RE = re.compile(r"\[((?:arxiv|pubmed):[^\]\s]+)\]")


class AgentError(RuntimeError):
    """Raised when an agent's model call fails (e.g. Ollama not running)."""


def extract_citations(text: str) -> List[str]:
    """Unique citation ids in order of appearance, e.g. ['arxiv:2401.00001']."""
    seen: List[str] = []
    for uid in CITATION_RE.findall(text):
        if uid not in seen:
            seen.append(uid)
    return seen


def split_citations(text: str, allowed: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return (valid, unknown). 'unknown' ids were cited but never retrieved -> possible hallucination."""
    allowed_set = set(allowed)
    cited = extract_citations(text)
    return [c for c in cited if c in allowed_set], [c for c in cited if c not in allowed_set]


def format_evidence(items: Sequence[RetrievedPaper], max_abstract_chars: int = 1200) -> str:
    """Render papers as delimited, citable blocks. Header line format is relied on by the stub + prompts."""
    blocks = []
    for item in items:
        p = item.paper
        abstract = p.abstract if len(p.abstract) <= max_abstract_chars else p.abstract[:max_abstract_chars] + " ..."
        blocks.append(
            f"[{p.uid}] {p.title}\n"
            f"Authors: {', '.join(p.authors[:5]) or 'n/a'} | Published: {p.published[:10] or 'n/a'} | URL: {p.url}\n"
            f"Abstract: {abstract}"
        )
    return "\n\n".join(blocks)
