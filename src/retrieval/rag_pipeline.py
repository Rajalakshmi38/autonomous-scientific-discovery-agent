"""RAG layer: retrieve evidence, fit it to a prompt budget, and check that answers are grounded in it.

What "grounded" means here (and what it does NOT mean):
  * every claim-bearing sentence should cite a retrieved paper by id, e.g. [arxiv:2401.00001];
  * ids that were never retrieved are flagged as likely fabrications;
  * a LEXICAL heuristic flags cited sentences that share few words with the cited papers.
This does not prove a claim is true or even supported; it only catches obvious problems.
A person still has to read the sources.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from src.agents.base import extract_citations, format_evidence
from src.models import RetrievedPaper
from src.retrieval.retriever import PaperRetriever

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z][a-z\-]{3,}")
_CITE_RE = re.compile(r"\[(?:arxiv|pubmed):[^\]\s]+\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = frozenset(
    "that this with from have has were which their there these those been being also such than then they them "
    "into over under about between within using used uses based paper papers study approach method methods "
    "results result show shows shown propose proposed present presented provide provides".split()
)
WEAK_SUPPORT_THRESHOLD = 0.25


@dataclass
class Source:
    id: str
    title: str
    url: str
    year: str = ""
    score: float = 0.0


@dataclass
class RagContext:
    topic: str
    evidence: List[RetrievedPaper]
    sources: List[Source]
    dropped: int = 0          # papers left out because they did not fit the prompt budget


@dataclass
class GroundingReport:
    cited_ids: List[str] = field(default_factory=list)
    unknown_ids: List[str] = field(default_factory=list)
    cited_sources: List[Source] = field(default_factory=list)       # traceable ids + urls
    weakly_supported: List[str] = field(default_factory=list)       # sentences with little lexical support
    grounded: bool = False                                          # >=1 valid citation and no unknown ids


def _words(text: str) -> set:
    return {w for w in _WORD_RE.findall(_CITE_RE.sub(" ", text).lower()) if w not in _STOP}


class RagPipeline:
    def __init__(self, retriever: PaperRetriever, char_budget: int = 6000, max_abstract_chars: int = 1200) -> None:
        self.retriever = retriever
        self.char_budget = char_budget
        self.max_abstract_chars = max_abstract_chars

    def prepare(self, topic: str, k: int = 5) -> RagContext:
        """Retrieve up to ``k`` papers, then keep the best-ranked ones that fit the character budget."""
        retrieved = self.retriever.retrieve(topic, k=k)
        kept: List[RetrievedPaper] = []
        used = 0
        for item in retrieved:
            size = len(format_evidence([item], self.max_abstract_chars))
            if kept and used + size > self.char_budget:
                break                      # always keep at least the top result
            kept.append(item)
            used += size
        dropped = len(retrieved) - len(kept)
        if dropped:
            logger.info("Dropped %d lower-ranked paper(s) to fit the prompt budget", dropped)
        sources = [Source(i.paper.uid, i.paper.title, i.paper.url, i.paper.year, round(i.score, 4)) for i in kept]
        return RagContext(topic=topic, evidence=kept, sources=sources, dropped=dropped)

    def check(self, text: str, context: RagContext) -> GroundingReport:
        """Check citations in ``text`` against the retrieved evidence."""
        by_id = {e.paper.uid: e.paper for e in context.evidence}
        source_by_id = {s.id: s for s in context.sources}
        cited = extract_citations(text)
        valid = [c for c in cited if c in by_id]
        unknown = [c for c in cited if c not in by_id]

        weak: List[str] = []
        for sentence in _SENTENCE_RE.split(text):
            ids = [c for c in extract_citations(sentence) if c in by_id]
            if not ids:
                continue
            words = _words(sentence)
            if len(words) < 4:
                continue
            support_words = set()
            for uid in ids:
                support_words |= _words(by_id[uid].title + " " + by_id[uid].abstract)
            if len(words & support_words) / len(words) < WEAK_SUPPORT_THRESHOLD:
                weak.append(sentence.strip()[:200])

        return GroundingReport(
            cited_ids=valid,
            unknown_ids=unknown,
            cited_sources=[source_by_id[c] for c in valid],
            weakly_supported=weak,
            grounded=bool(valid) and not unknown,
        )
