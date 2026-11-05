"""arXiv ingestion via the public Atom API (https://info.arxiv.org/help/api/).

Uses only ``requests`` + the standard-library XML parser. Retrieved text is
treated as UNTRUSTED data throughout the project.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, List, Optional

import requests

from src.ingestion.base import IngestionError, PaperSource
from src.models import Paper

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS_PER_CALL = 100   # kept modest on purpose; arXiv asks for polite usage

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v(?P<ver>\d+))?$")
_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\+\.]*")
_CATEGORY_RE = re.compile(r"^[A-Za-z\-]+(\.[A-Za-z\-]+)?$")
_RETRY_STATUS = {429, 500, 502, 503, 504}
# Filler words are dropped from the search, because every remaining word must match and "of" or "the" only
# make a query return nothing. If a query is made ONLY of such words, they are all kept.
_FILLER_WORDS = {"a", "an", "the", "of", "in", "on", "for", "to", "with", "by", "from", "at", "is", "are",
                 "as", "and", "or", "via", "using", "into"}


def build_search_query(topic: str, category: Optional[str] = None) -> str:
    """Turn free text into an arXiv ``search_query`` (all terms must match; filler words such as "of" are dropped)."""
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Search topic must be a non-empty string.")
    terms = _TERM_RE.findall(topic)
    if not terms:
        raise ValueError(f"No searchable words found in topic: {topic!r}")
    content_terms = [t for t in terms if t.lower() not in _FILLER_WORDS]
    terms = content_terms or terms
    query = " AND ".join(f"all:{t}" for t in terms)
    if category:
        if not _CATEGORY_RE.match(category):
            raise ValueError(f"Invalid arXiv category: {category!r} (example: cs.LG)")
        query += f" AND cat:{category}"
    return query


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text).strip() if text else ""


def _find_text(entry: ET.Element, path: str) -> str:
    node = entry.find(path, NS)
    return _clean(node.text) if node is not None else ""


def parse_arxiv_feed(xml_text: str) -> List[Paper]:
    """Parse an arXiv Atom response into Paper objects."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise IngestionError(f"arXiv returned XML that could not be parsed: {exc}") from exc

    papers: List[Paper] = []
    for entry in root.findall("atom:entry", NS):
        entry_id = _find_text(entry, "atom:id")
        if "/api/errors" in entry_id:   # arXiv reports query errors as a fake entry
            raise IngestionError(f"arXiv API error: {_find_text(entry, 'atom:summary')}")
        match = _ID_RE.search(entry_id)
        if not match:
            logger.warning("Skipping entry with unrecognised id: %r", entry_id)
            continue

        paper_id = match.group("id")
        url, pdf_url = f"https://arxiv.org/abs/{paper_id}", ""
        for link in entry.findall("atom:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
            elif link.get("rel") == "alternate":
                url = link.get("href", url)

        papers.append(
            Paper(
                source="arxiv",
                paper_id=paper_id,
                title=_find_text(entry, "atom:title"),
                abstract=_find_text(entry, "atom:summary"),
                authors=[
                    _clean(a.findtext("atom:name", default="", namespaces=NS))
                    for a in entry.findall("atom:author", NS)
                ],
                published=_find_text(entry, "atom:published"),
                updated=_find_text(entry, "atom:updated"),
                url=url,
                pdf_url=pdf_url,
                categories=[c.get("term", "") for c in entry.findall("atom:category", NS)],
                doi=_find_text(entry, "arxiv:doi") or None,
                journal_ref=_find_text(entry, "arxiv:journal_ref") or None,
                version=int(match.group("ver")) if match.group("ver") else None,
            )
        )
    return papers


class ArxivClient(PaperSource):
    name = "arxiv"

    def __init__(
        self,
        timeout: float = 30.0,
        retries: int = 3,
        delay_seconds: float = 3.0,
        contact: str = "",
        raw_dir: Optional[Path] = None,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout = timeout
        self.retries = max(1, retries)
        self.delay_seconds = delay_seconds
        self.raw_dir = raw_dir
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last_request: Optional[float] = None
        agent = "autonomous-scientific-discovery-agent/0.1 (student research project"
        self.user_agent = agent + (f"; contact: {contact})" if contact else ")")

    def _wait_for_rate_limit(self) -> None:
        if self._last_request is None:
            return
        remaining = self.delay_seconds - (time.monotonic() - self._last_request)
        if remaining > 0:
            self._sleep(remaining)

    def _fetch(self, params: dict) -> str:
        last_error: object = None
        for attempt in range(1, self.retries + 1):
            try:
                self._wait_for_rate_limit()
                response = self.session.get(
                    ARXIV_API_URL,
                    params=params,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                )
                self._last_request = time.monotonic()
                if response.status_code == 200:
                    return response.text
                if response.status_code not in _RETRY_STATUS:
                    raise IngestionError(f"arXiv returned HTTP {response.status_code}")
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = exc
            logger.warning("arXiv attempt %d/%d failed: %s", attempt, self.retries, last_error)
            if attempt < self.retries:
                self._sleep(min(2 ** attempt, 30))
        raise IngestionError(f"arXiv request failed after {self.retries} attempts: {last_error}")

    def search(
        self,
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
        start: int = 0,
    ) -> List[Paper]:
        if not 1 <= max_results <= MAX_RESULTS_PER_CALL:
            raise ValueError(f"max_results must be between 1 and {MAX_RESULTS_PER_CALL}")
        if start < 0:
            raise ValueError("start must be >= 0")

        params = {
            "search_query": build_search_query(query, category),
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        logger.info("Querying arXiv: %s", params["search_query"])
        xml_text = self._fetch(params)
        self._save_raw(xml_text)
        papers = parse_arxiv_feed(xml_text)
        logger.info("arXiv returned %d paper(s)", len(papers))
        return papers

    def _save_raw(self, xml_text: str) -> None:
        if self.raw_dir is None:
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"arxiv_{time.strftime('%Y%m%d_%H%M%S')}.xml"
        path.write_text(xml_text, encoding="utf-8")
        logger.debug("Saved raw arXiv response to %s", path)
