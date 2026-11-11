"""PubMed ingestion, the watch list and the update command. The XML below is TEST DATA written by hand in the shape
of NCBI's answers; it is not a real record."""
import json

import pytest
import requests

from config.settings import load_settings
from src.ingestion import watchlist
from src.ingestion.base import IngestionError
from src.ingestion.paper_store import PaperStore
from src.ingestion.pubmed_client import PubMedClient, parse_pubmed_xml
from src.main import build_retriever, main
from src.models import Paper

XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation><PMID Version="1">111</PMID>
   <Article>
    <Journal><Title>Journal of Tests</Title><JournalIssue><PubDate><Year>2026</Year><Month>Jan</Month><Day>15</Day></PubDate></JournalIssue></Journal>
    <ArticleTitle>Effect of <i>feature</i> scaling on  diagnosis</ArticleTitle>
    <Abstract><AbstractText Label="BACKGROUND">Scaling matters.</AbstractText><AbstractText Label="RESULTS">Recall rose from 0.5 to 0.8.</AbstractText></Abstract>
    <AuthorList><Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author><Author><CollectiveName>The Test Group</CollectiveName></Author></AuthorList>
    <ArticleDate DateType="Electronic"><Year>2026</Year><Month>01</Month><Day>10</Day></ArticleDate>
   </Article></MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1000/test.111</ArticleId></ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>222</PMID>
   <Article><Journal><Title>Other Journal</Title><JournalIssue><PubDate><MedlineDate>2025 Nov-Dec</MedlineDate></PubDate></JournalIssue></Journal>
    <ArticleTitle>A paper with a plain abstract</ArticleTitle>
    <Abstract><AbstractText>Just one paragraph.</AbstractText></Abstract></Article></MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>333</PMID><Article><ArticleTitle>No abstract here</ArticleTitle></Article></MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>"""


class FakeResponse:
    def __init__(self, status=200, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        if self._data is None:
            raise ValueError("not json")
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(responses, **kw):
    return PubMedClient(session=FakeSession(responses), sleep=lambda s: None, delay_seconds=0, **kw)


# ------------------------------------------------------------------ parsing
def test_records_are_parsed_with_labels_inline_tags_dates_authors_and_doi():
    papers = parse_pubmed_xml(XML)
    assert [p.paper_id for p in papers] == ["111", "222"]                 # 333 has no abstract and is skipped
    first = papers[0]
    assert first.source == "pubmed" and first.uid == "pubmed:111"
    assert first.title == "Effect of feature scaling on diagnosis"
    assert first.abstract == "Background: Scaling matters. Results: Recall rose from 0.5 to 0.8."
    assert first.authors == ["Jane Smith", "The Test Group"]
    assert first.published == "2026-01-10T00:00:00Z" and first.year == "2026"       # ArticleDate wins over PubDate
    assert first.doi == "10.1000/test.111" and first.journal_ref == "Journal of Tests"
    assert first.url == "https://pubmed.ncbi.nlm.nih.gov/111/"
    assert papers[1].published == "2025-11-01T00:00:00Z" and papers[1].doi is None   # MedlineDate


def test_bad_xml_is_reported_as_an_ingestion_error():
    with pytest.raises(IngestionError):
        parse_pubmed_xml("<not xml")


# ------------------------------------------------------------------ talking to NCBI
def test_search_finds_ids_then_fetches_the_records():
    c = client([FakeResponse(data={"esearchresult": {"idlist": ["111", "222"]}}), FakeResponse(text=XML)], contact="me@example.org")
    papers = c.search("feature scaling", max_results=5)
    assert len(papers) == 2
    (url1, p1), (url2, p2) = c.session.calls
    assert "esearch" in url1 and p1["term"] == "feature scaling" and p1["retmax"] == 5 and p1["email"] == "me@example.org"
    assert "efetch" in url2 and p2["id"] == "111,222"


def test_no_hits_means_no_second_call():
    c = client([FakeResponse(data={"esearchresult": {"idlist": []}})])
    assert c.search("zzzz") == [] and len(c.session.calls) == 1


def test_temporary_errors_are_retried_and_permanent_ones_are_not():
    ok = FakeResponse(data={"esearchresult": {"idlist": []}})
    assert client([FakeResponse(503), requests.ConnectionError("down"), ok], retries=3).search("x") == []
    with pytest.raises(IngestionError, match="HTTP 400"):
        client([FakeResponse(400)]).search("x")
    with pytest.raises(IngestionError, match="after 2 attempts"):
        client([FakeResponse(500), FakeResponse(500)], retries=2).search("x")


def test_unexpected_answers_and_bad_arguments_are_rejected():
    with pytest.raises(IngestionError):
        client([FakeResponse(data={"unexpected": 1})]).search("x")
    with pytest.raises(IngestionError):
        client([FakeResponse(text="not json")]).search("x")
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            client([]).search(bad)
    with pytest.raises(ValueError):
        client([]).search("x", max_results=0)


# ------------------------------------------------------------------ the watch list
def test_watch_list_add_list_remove_and_validation(tmp_path):
    path = tmp_path / "watch.json"
    assert watchlist.load(path) == []
    assert watchlist.add(path, watchlist.WatchEntry("arxiv", "random forest", 10, "cs.LG")) is True
    assert watchlist.add(path, watchlist.WatchEntry("arxiv", "random forest", 10, "cs.LG")) is False      # duplicate
    assert watchlist.add(path, watchlist.WatchEntry("pubmed", "asthma sounds")) is True
    assert [e.source for e in watchlist.load(path)] == ["arxiv", "pubmed"]
    assert watchlist.remove(path, 1).query == "random forest" and len(watchlist.load(path)) == 1
    for bad in (watchlist.WatchEntry("web", "x"), watchlist.WatchEntry("arxiv", " "), watchlist.WatchEntry("arxiv", "x", 0),
                watchlist.WatchEntry("pubmed", "x", 5, "cs.LG")):
        with pytest.raises(ValueError):
            watchlist.add(path, bad)
    with pytest.raises(ValueError, match="no entry 9"):
        watchlist.remove(path, 9)


def test_a_damaged_watch_list_is_explained(tmp_path):
    path = tmp_path / "watch.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="damaged"):
        watchlist.load(path)


# ------------------------------------------------------------------ the commands
def paper(pid, source="arxiv", title="T"):
    return Paper(source=source, paper_id=pid, title=title, abstract=f"abstract about {title}", published="2026-11-01T00:00:00Z",
                 url=f"https://example.org/{pid}")


class FakeSource:
    def __init__(self, papers=None, error=None):
        self.papers, self.error, self.calls = papers or [], error, []

    def search(self, query, max_results=10, category=None):
        self.calls.append((query, max_results, category))
        if self.error:
            raise IngestionError(self.error)
        return self.papers


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    return load_settings()


def test_ingest_from_pubmed_stores_and_indexes_the_papers(env, monkeypatch, capsys):
    fake = FakeSource([paper("111", "pubmed", "asthma sounds")])
    monkeypatch.setattr("src.main._make_source", lambda settings, name: fake)
    assert main(["ingest", "--source", "pubmed", "--query", "asthma"]) == 0
    assert "pubmed:111" in capsys.readouterr().out and fake.calls == [("asthma", 10, None)]
    store = PaperStore(env.papers_db_path)
    assert store.get("pubmed:111") is not None
    store.close()
    assert main(["ingest", "--source", "pubmed", "--query", "asthma", "--category", "cs.LG"]) == 1


def test_watch_commands_and_update_add_only_new_papers(env, monkeypatch, capsys):
    assert main(["update"]) == 0 and "watch list is empty" in capsys.readouterr().out
    assert main(["watch", "add", "--query", "random forest", "--category", "cs.LG", "--max-results", "5"]) == 0
    assert main(["watch", "add", "--source", "pubmed", "--query", "asthma"]) == 0
    assert main(["watch", "add", "--source", "pubmed", "--query", "x", "--category", "cs.LG"]) == 1      # not allowed
    capsys.readouterr()
    assert main(["watch", "list"]) == 0
    listing = capsys.readouterr().out
    assert "1. arxiv" in listing and "2. pubmed" in listing

    sources = {"arxiv": FakeSource([paper("1"), paper("2")]), "pubmed": FakeSource([paper("9", "pubmed")])}
    monkeypatch.setattr("src.main._make_source", lambda settings, name: sources[name])
    assert main(["update"]) == 0
    assert "Update finished: 3 new paper(s), 0 search(es) failed." in capsys.readouterr().out
    assert sources["arxiv"].calls == [("random forest", 5, "cs.LG")]

    sources["arxiv"].papers = [paper("1"), paper("2"), paper("3")]                  # one new paper appeared
    assert main(["update"]) == 0
    assert "Update finished: 1 new paper(s)" in capsys.readouterr().out
    store, retriever = build_retriever(env)
    assert store.count() == 4
    store.close()
    assert main(["watch", "remove", "2"]) == 0 and main(["watch", "remove", "9"]) == 1


def test_update_keeps_going_when_one_search_fails_and_says_so(env, monkeypatch, capsys):
    main(["watch", "add", "--query", "a"])
    main(["watch", "add", "--source", "pubmed", "--query", "b"])
    sources = {"arxiv": FakeSource(error="arXiv is down"), "pubmed": FakeSource([paper("5", "pubmed")])}
    monkeypatch.setattr("src.main._make_source", lambda settings, name: sources[name])
    capsys.readouterr()
    assert main(["update"]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and "arXiv is down" in out and "1 new paper(s), 1 search(es) failed" in out
