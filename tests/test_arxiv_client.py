"""Tests for arXiv ingestion. The fixture XML below is HAND-WRITTEN TEST DATA in
arXiv's Atom format; the papers in it are fictional and not real research."""
import os

import pytest
import requests

from src.ingestion.arxiv_client import ArxivClient, build_search_query, parse_arxiv_feed
from src.ingestion.base import IngestionError
from src.ingestion.paper_store import PaperStore

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v2</id>
    <updated>2024-02-01T10:00:00Z</updated>
    <published>2024-01-02T09:00:00Z</published>
    <title>Fictional Test Paper
      On Graph Learning</title>
    <summary>  A fictional abstract
      about graph learning.  </summary>
    <author><name>Test Author One</name></author>
    <author><name>Test Author Two</name></author>
    <arxiv:doi>10.0000/fictional</arxiv:doi>
    <link href="http://arxiv.org/abs/2401.00001v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v2" rel="related" type="application/pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0601001v1</id>
    <published>2006-01-01T00:00:00Z</published>
    <title>Old Style Id Paper</title>
    <summary>Abstract two.</summary>
    <author><name>Someone</name></author>
  </entry>
</feed>"""

EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
ERROR_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/api/errors#bad</id><title>Error</title>
<summary>incorrect id format</summary></entry></feed>"""


class FakeResponse:
    def __init__(self, text="", status_code=200):
        self.text, self.status_code = text, status_code


class FakeSession:
    """Returns queued responses/exceptions in order; records calls."""

    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes, **kw):
    return ArxivClient(session=FakeSession(outcomes), sleep=lambda s: None, delay_seconds=0, **kw)


# ---------------------------------------------------------------- parsing
def test_parse_extracts_all_fields():
    papers = parse_arxiv_feed(SAMPLE_FEED)
    assert len(papers) == 2
    p = papers[0]
    assert p.uid == "arxiv:2401.00001"
    assert p.version == 2
    assert p.title == "Fictional Test Paper On Graph Learning"      # whitespace collapsed
    assert p.abstract == "A fictional abstract about graph learning."
    assert p.authors == ["Test Author One", "Test Author Two"]
    assert p.year == "2024"
    assert p.doi == "10.0000/fictional"
    assert p.url == "http://arxiv.org/abs/2401.00001v2"
    assert p.pdf_url == "http://arxiv.org/pdf/2401.00001v2"
    assert p.categories == ["cs.LG"]


def test_parse_old_style_id():
    assert parse_arxiv_feed(SAMPLE_FEED)[1].paper_id == "cs/0601001"


def test_parse_empty_feed_returns_empty_list():
    assert parse_arxiv_feed(EMPTY_FEED) == []


def test_parse_api_error_entry_raises():
    with pytest.raises(IngestionError, match="incorrect id format"):
        parse_arxiv_feed(ERROR_FEED)


def test_parse_malformed_xml_raises():
    with pytest.raises(IngestionError):
        parse_arxiv_feed("<feed><not-closed>")


# ---------------------------------------------------------------- queries
def test_build_query_combines_terms_and_category():
    assert build_search_query("graph neural networks", "cs.LG") == (
        "all:graph AND all:neural AND all:networks AND cat:cs.LG"
    )


@pytest.mark.parametrize("bad", ["", "   ", "!!! ???", None])
def test_build_query_rejects_empty_input(bad):
    with pytest.raises(ValueError):
        build_search_query(bad)


def test_build_query_rejects_bad_category():
    with pytest.raises(ValueError):
        build_search_query("x", "cs.LG OR all:evil")


# ---------------------------------------------------------------- client
def test_search_success_and_sends_user_agent():
    client = make_client([FakeResponse(SAMPLE_FEED)], contact="me@example.com")
    papers = client.search("graph learning", max_results=2)
    assert len(papers) == 2
    _, params, headers, _ = client.session.calls[0]
    assert params["max_results"] == 2
    assert "me@example.com" in headers["User-Agent"]


def test_search_no_results_is_empty_list():
    assert make_client([FakeResponse(EMPTY_FEED)]).search("anything") == []


def test_retries_then_succeeds():
    client = make_client([requests.ConnectionError("boom"), FakeResponse("", 503), FakeResponse(SAMPLE_FEED)])
    assert len(client.search("graph")) == 2
    assert len(client.session.calls) == 3


def test_network_down_raises_ingestion_error():
    client = make_client([requests.ConnectionError("down")] * 3, retries=3)
    with pytest.raises(IngestionError, match="failed after 3 attempts"):
        client.search("graph")


def test_non_retryable_http_status_fails_fast():
    client = make_client([FakeResponse("", 400)])
    with pytest.raises(IngestionError, match="HTTP 400"):
        client.search("graph")
    assert len(client.session.calls) == 1


@pytest.mark.parametrize("n", [0, 101])
def test_search_validates_max_results(n):
    with pytest.raises(ValueError):
        make_client([]).search("graph", max_results=n)


def test_raw_response_is_saved(tmp_path):
    make_client([FakeResponse(SAMPLE_FEED)], raw_dir=tmp_path).search("graph")
    assert len(list(tmp_path.glob("arxiv_*.xml"))) == 1


# ---------------------------------------------------------------- storage
def test_paper_store_roundtrip_and_upsert(tmp_path):
    store = PaperStore(tmp_path / "p.db")
    try:
        papers = parse_arxiv_feed(SAMPLE_FEED)
        store.upsert_many(papers)
        store.upsert_many(papers)                       # no duplicates
        assert store.count() == 2
        assert store.get("arxiv:2401.00001").authors == ["Test Author One", "Test Author Two"]
        assert store.get("arxiv:missing") is None
    finally:
        store.close()


# ---------------------------------------------------------------- live (opt-in)
@pytest.mark.network
@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1 to hit real arXiv")
def test_live_arxiv_search():
    papers = ArxivClient().search("graph neural networks", max_results=3)
    assert 1 <= len(papers) <= 3
    assert all(p.title and p.url for p in papers)
