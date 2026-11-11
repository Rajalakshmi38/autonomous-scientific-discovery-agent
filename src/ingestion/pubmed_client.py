"""PubMed ingestion through NCBI E-utilities (esearch to find ids, efetch to get the records).

Uses only ``requests`` and the standard-library XML parser. Retrieved text is UNTRUSTED data, as for arXiv.
Papers without an abstract are skipped, because there is nothing to search or summarize in them.
NCBI allows about three requests per second without an API key; the client waits between calls.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Callable, List, Optional

import requests

from src.ingestion.base import IngestionError, PaperSource
from src.models import Paper

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MAX_RESULTS_PER_CALL = 100
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _text(node: Optional[ET.Element]) -> str:
    """All text inside an element, including inline tags such as <i>, with whitespace collapsed."""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip() if node is not None else ""


def _month(value: str) -> int:
    value = (value or "").strip().lower()
    if value.isdigit() and 1 <= int(value) <= 12:
        return int(value)
    return _MONTHS.get(value[:3], 1)


def _date(article: ET.Element) -> str:
    """ISO timestamp (like the arXiv records) from ArticleDate, else the journal's PubDate, else ''."""
    node = article.find("ArticleDate")
    if node is None:
        node = article.find("Journal/JournalIssue/PubDate")
    if node is None:
        return ""
    year = _text(node.find("Year"))
    if not year:                                        # e.g. <MedlineDate>2026 Jan-Feb</MedlineDate>
        found = re.search(r"(\d{4})(?:\s+([A-Za-z]{3}))?", _text(node.find("MedlineDate")))
        if not found:
            return ""
        year, month, day = found.group(1), _month(found.group(2) or ""), 1
    else:
        month, day = _month(_text(node.find("Month"))), int(_text(node.find("Day")) or 1) if _text(node.find("Day")).isdigit() else 1
    return f"{int(year):04d}-{month:02d}-{day:02d}T00:00:00Z"


def parse_pubmed_xml(xml_text: str) -> List[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise IngestionError(f"PubMed returned data that is not valid XML: {exc}") from exc
    papers: List[Paper] = []
    skipped = 0
    for item in root.findall("PubmedArticle"):
        citation = item.find("MedlineCitation")
        article = citation.find("Article") if citation is not None else None
        pmid = _text(citation.find("PMID")) if citation is not None else ""
        if article is None or not pmid:
            continue
        parts = []
        for node in article.findall("Abstract/AbstractText"):
            label, body = node.get("Label"), _text(node)
            if body:
                parts.append(f"{label.title()}: {body}" if label else body)
        abstract = " ".join(parts)
        title = _text(article.find("ArticleTitle"))
        if not abstract or not title:
            skipped += 1
            continue
        authors = []
        for a in article.findall("AuthorList/Author"):
            name = _text(a.find("CollectiveName")) or " ".join(x for x in (_text(a.find("ForeName")), _text(a.find("LastName"))) if x)
            if name:
                authors.append(name)
        doi = next((_text(i) for i in item.findall("PubmedData/ArticleIdList/ArticleId") if i.get("IdType") == "doi"), None)
        papers.append(Paper(source="pubmed", paper_id=pmid, title=title, abstract=abstract, authors=authors,
                            published=_date(article), url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            doi=doi or None, journal_ref=_text(article.find("Journal/Title")) or None))
    if skipped:
        logger.info("Skipped %d PubMed record(s) without an abstract", skipped)
    return papers


class PubMedClient(PaperSource):
    name = "pubmed"

    def __init__(self, timeout: float = 30.0, retries: int = 3, delay_seconds: float = 0.4, contact: str = "",
                 session: Optional[requests.Session] = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.timeout = timeout
        self.retries = max(1, retries)
        self.delay_seconds = delay_seconds
        self.contact = contact
        self.session = session or requests.Session()
        self._sleep = sleep
        self._last_request: Optional[float] = None
        agent = "autonomous-scientific-discovery-agent/0.1 (student research project"
        self.user_agent = agent + (f"; contact: {contact})" if contact else ")")

    def _get(self, url: str, params: dict) -> requests.Response:
        params = dict(params, tool="autonomous-scientific-discovery-agent")
        if self.contact:
            params["email"] = self.contact
        last_error: object = None
        for attempt in range(1, self.retries + 1):
            try:
                if self._last_request is not None:
                    wait = self.delay_seconds - (time.monotonic() - self._last_request)
                    if wait > 0:
                        self._sleep(wait)
                response = self.session.get(url, params=params, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
                self._last_request = time.monotonic()
                if response.status_code == 200:
                    return response
                if response.status_code not in _RETRY_STATUS:
                    raise IngestionError(f"PubMed returned HTTP {response.status_code}")
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = exc
            logger.warning("PubMed attempt %d/%d failed: %s", attempt, self.retries, last_error)
            if attempt < self.retries:
                self._sleep(min(2 ** attempt, 30))
        raise IngestionError(f"PubMed request failed after {self.retries} attempts: {last_error}")

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if not 1 <= max_results <= MAX_RESULTS_PER_CALL:
            raise ValueError(f"max_results must be between 1 and {MAX_RESULTS_PER_CALL}")
        logger.info("Querying PubMed: %s", query)
        found = self._get(ESEARCH_URL, {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"})
        try:
            ids = [str(i) for i in found.json()["esearchresult"]["idlist"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise IngestionError(f"PubMed search returned an unexpected answer: {exc}") from exc
        if not ids:
            return []
        records = self._get(EFETCH_URL, {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
        papers = parse_pubmed_xml(records.text)
        logger.info("PubMed returned %d paper(s) with an abstract", len(papers))
        return papers
