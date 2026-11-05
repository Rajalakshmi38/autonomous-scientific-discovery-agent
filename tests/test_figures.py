"""Figure extraction tests. The PDFs are generated here from MADE-UP text; no real paper is used."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("pypdf")
pytest.importorskip("pypdfium2")
PIL = pytest.importorskip("PIL.Image")

from src.ingestion.figures import (
    FigureError, FigureInfo, describe_figure, download_pdf, extract_figures, find_captions, load_manifest, paper_dir,
    save_manifest,
)
from src.ingestion.paper_store import PaperStore
from src.main import main
from src.models import Paper
from src.retrieval.embeddings import HashingEmbedder
from src.retrieval.rag_pipeline import RagPipeline
from src.retrieval.retriever import PaperRetriever
from src.retrieval.vector_store import NumpyVectorStore
from tests.pdf_helper import make_pdf

PAGE1 = ["Abstract. We compare two methods.", "As shown in Figure 1 the method works well on wine.", "Body text."]
PAGE2 = ["Figure 1: Accuracy of method A and", "method B on the wine dataset.", "", "Figure 2. Training loss over epochs.", "", "More text."]


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf([PAGE1, PAGE2]))
    return path


# ------------------------------------------------------------------ caption detection
def test_only_captions_that_start_a_line_are_found():
    text = "Intro\nAs shown in Figure 2: we see.\nFigure 1: Accuracy of A and B.\n\nFig. 3. Loss curves over time.\n"
    assert find_captions(text) == [("Figure 1", "Accuracy of A and B."), ("Figure 3", "Loss curves over time.")]


def test_multi_line_captions_are_joined_and_short_junk_is_ignored():
    assert find_captions("Figure 4: A caption that\ncontinues here.\n\nFigure 5: ok\n") == [("Figure 4", "A caption that continues here.")]


# ------------------------------------------------------------------ extraction from a real PDF
def test_figures_are_found_rendered_and_saved(pdf_file, tmp_path):
    out = tmp_path / "figs"
    figures = extract_figures(pdf_file, out)
    assert [(f.page, f.label) for f in figures] == [(2, "Figure 1"), (2, "Figure 2")]
    assert "wine dataset" in figures[0].caption
    assert figures[1].caption.startswith("Training loss over epochs.")     # may carry a few trailing words: captions are found heuristically
    with PIL.open(figures[0].image_path) as image:
        assert image.width > 300 and image.height > 300
    assert load_manifest(out) == figures


def test_a_pdf_without_captions_gives_an_empty_result(tmp_path):
    path = tmp_path / "plain.pdf"
    path.write_bytes(make_pdf([["Just text, nothing else to see here."]]))
    assert extract_figures(path, tmp_path / "o") == []


def test_the_number_of_figures_is_capped(pdf_file, tmp_path):
    assert len(extract_figures(pdf_file, tmp_path / "o", max_figures=1)) == 1


def test_a_broken_pdf_is_reported_not_crashed(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf")
    with pytest.raises(FigureError):
        extract_figures(path, tmp_path / "o")


def test_paper_folder_names_are_safe(tmp_path):
    assert paper_dir(tmp_path, "arxiv:cs/0601001").name == "arxiv_cs_0601001"
    hostile = paper_dir(tmp_path, "arxiv:../../etc/passwd")
    assert hostile.parent == tmp_path and "/" not in hostile.name and hostile.name not in (".", "..")


def test_missing_or_damaged_manifests_are_tolerated(tmp_path):
    assert load_manifest(tmp_path) == []
    (tmp_path / "figures.json").write_text("{broken", encoding="utf-8")
    assert load_manifest(tmp_path) == []


# ------------------------------------------------------------------ downloading
class FakeResponse:
    def __init__(self, content=b"%PDF-1.4 fake", status=200, url="https://arxiv.org/pdf/1", length=None):
        self.content, self.status_code, self.url = content, status, url
        self.headers = {"Content-Length": str(length)} if length is not None else {}

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class FakeSession:
    def __init__(self, response):
        self.response, self.urls = response, []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.response


def test_download_upgrades_http_to_https_and_saves_the_file(tmp_path):
    session = FakeSession(FakeResponse())
    path = download_pdf("http://arxiv.org/pdf/2610.21958v1", tmp_path / "x" / "paper.pdf", session=session)
    assert session.urls == ["https://arxiv.org/pdf/2610.21958v1"] and path.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("url", ["https://evil.example.com/a.pdf", "ftp://arxiv.org/a.pdf", "file:///etc/passwd", "https://arxiv.org.evil.com/a.pdf"])
def test_only_arxiv_is_allowed(url, tmp_path):
    with pytest.raises(FigureError, match="only arxiv.org"):
        download_pdf(url, tmp_path / "p.pdf", session=FakeSession(FakeResponse()))


def test_redirects_away_from_arxiv_are_refused(tmp_path):
    with pytest.raises(FigureError, match="redirected"):
        download_pdf("https://arxiv.org/pdf/1", tmp_path / "p.pdf", session=FakeSession(FakeResponse(url="https://evil.example.com/x")))


@pytest.mark.parametrize("response,fragment", [
    (FakeResponse(content=b"<html>not a pdf</html>"), "not a PDF"),
    (FakeResponse(status=404), "HTTP 404"),
    (FakeResponse(length=10 ** 9), "larger"),
    (FakeResponse(content=b"%PDF" + b"x" * 5000), "larger"),
])
def test_bad_downloads_are_rejected(response, fragment, tmp_path):
    with pytest.raises(FigureError, match=fragment):
        download_pdf("https://arxiv.org/pdf/1", tmp_path / "p.pdf", session=FakeSession(response), max_bytes=1000)
    assert not (tmp_path / "p.pdf").exists()


# ------------------------------------------------------------------ optional vision model (against a FAKE Ollama)
class FakeVision(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(req)
        status, body = (404, {"error": "model not found"}) if req["model"] == "missing" else (
            200, {"message": {"role": "assistant", "content": "A bar chart comparing two methods."}, "done": True})
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def vision_server(monkeypatch):
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVision)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield server
    server.shutdown()
    server.server_close()


def test_description_request_carries_the_image_and_caption(pdf_file, tmp_path, vision_server):
    figure = extract_figures(pdf_file, tmp_path / "o")[0]
    text = describe_figure(figure.image_path, figure.caption, vision_server.url, "tiny-vision")
    assert text == "A bar chart comparing two methods."
    req = vision_server.requests[0]
    assert req["model"] == "tiny-vision" and len(req["messages"][0]["images"]) == 1 and "wine dataset" in req["messages"][0]["content"]


def test_vision_errors_are_clear(pdf_file, tmp_path, vision_server):
    image = extract_figures(pdf_file, tmp_path / "o")[0].image_path
    with pytest.raises(FigureError, match="vision model"):
        describe_figure(image, "c", vision_server.url, "missing")
    with pytest.raises(FigureError, match="No vision model"):
        describe_figure(image, "c", vision_server.url, "")
    with pytest.raises(FigureError, match="Could not reach"):
        describe_figure(image, "c", "http://127.0.0.1:1", "x", timeout=2)


# ------------------------------------------------------------------ captions reach the evidence
def test_cached_captions_are_added_to_the_evidence(tmp_path):
    paper = Paper("arxiv", "1", "Graph neural networks", "Message passing on graph neural networks.", url="u1")
    store = PaperStore(tmp_path / "p.db")
    store.upsert_many([paper])
    retriever = PaperRetriever(store, HashingEmbedder(), NumpyVectorStore())
    retriever.index_papers([paper])
    folder = paper_dir(tmp_path / "figures", paper.uid)
    folder.mkdir(parents=True)
    save_manifest(folder, [FigureInfo(3, "Figure 2", "Accuracy by depth.", "x.png")])
    try:
        context = RagPipeline(retriever, figures_dir=tmp_path / "figures").prepare("graph neural networks", k=3)
        assert context.evidence[0].captions == ["Figure 2 (page 3): Accuracy by depth."]
        from src.agents.base import format_evidence
        assert "Figure captions: Figure 2 (page 3): Accuracy by depth." in format_evidence(context.evidence)
        plain = RagPipeline(retriever).prepare("graph neural networks", k=3)
        assert plain.evidence[0].captions == []
    finally:
        store.close()


# ------------------------------------------------------------------ the command
def test_figures_command_end_to_end(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "s"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    store = PaperStore(tmp_path / "s" / "papers.db")
    store.upsert_many([Paper("arxiv", "1", "T", "A", pdf_url="https://arxiv.org/pdf/1")])
    store.close()

    def fake_download(url, dest, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(make_pdf([PAGE1, PAGE2]))
        return dest
    monkeypatch.setattr("src.main.download_pdf", fake_download)
    assert main(["figures", "--paper", "arxiv:1"]) == 0
    out = capsys.readouterr().out
    assert "Figure 1 (page 2)" in out and "wine dataset" in out
    assert load_manifest(paper_dir(tmp_path / "d" / "figures", "arxiv:1"))
    assert main(["figures", "--paper", "arxiv:999"]) == 1
    assert "not stored" in capsys.readouterr().out


def test_find_captions_drops_a_page_number_under_the_caption():
    text = "Figure 2: The analysis view, showing a map.\n7\n"
    assert find_captions(text) == [("Figure 2", "The analysis view, showing a map.")]


def test_find_captions_keeps_a_number_that_ends_the_sentence():
    assert find_captions("Figure 6: Loss after epoch 50\n") == [("Figure 6", "Loss after epoch 50")]
