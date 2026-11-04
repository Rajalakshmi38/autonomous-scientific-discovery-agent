"""Figures from research-paper PDFs (the multimodal part of the project).

What it does: downloads a paper's PDF from arXiv, finds figure captions in the text, renders the pages that
hold figures to PNG images, and saves a manifest. A vision-language model can OPTIONALLY describe each page.

What it does NOT do (be honest about this):
  * It does not crop individual charts. It renders the whole PAGE that holds a figure caption.
  * It does not read numbers from charts. A model's description of a figure is unverified text.
  * Papers are untrusted input. Only arxiv.org PDFs are fetched, size and type are checked, and pages are
    limited, but PDF parsing runs on the host (not in the sandbox).
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

ALLOWED_HOSTS = {"arxiv.org", "export.arxiv.org"}
MAX_PDF_BYTES = 25 * 1024 * 1024
MANIFEST = "figures.json"
_CAPTION_START = re.compile(r"^\s*(?:Figure|Fig\.|FIGURE)\s*(\d{1,3})\s*[.:]\s*", re.MULTILINE)


class FigureError(RuntimeError):
    """A PDF could not be downloaded, read or rendered."""


@dataclass
class FigureInfo:
    page: int                      # 1-based page number
    label: str                     # e.g. "Figure 3"
    caption: str
    image_path: str                # PNG of the whole page that holds the caption
    description: str = ""          # optional, model-written, UNVERIFIED
    description_source: str = ""


def paper_dir(figures_root: Path, uid: str) -> Path:
    """uid such as 'arxiv:2610.21958' or 'arxiv:cs/0601001' -> a safe folder name."""
    return Path(figures_root) / re.sub(r"[^A-Za-z0-9._-]", "_", uid)


def download_pdf(url: str, dest: Path, session: Optional[requests.Session] = None,
                 timeout: float = 60.0, max_bytes: int = MAX_PDF_BYTES) -> Path:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS or parsed.scheme not in ("http", "https"):
        raise FigureError(f"Refusing to download from {host or url!r}: only arxiv.org PDFs are allowed.")
    url = "https://" + url.split("://", 1)[1]            # arXiv records store http:// links; always fetch over https
    http = session or requests
    try:
        resp = http.get(url, stream=True, timeout=timeout,
                        headers={"User-Agent": "autonomous-scientific-discovery-agent/0.1 (student research project)"})
    except requests.RequestException as exc:
        raise FigureError(f"Could not download the PDF: {exc}") from exc
    final_host = (urlparse(str(getattr(resp, "url", url))).hostname or "").lower()
    if final_host not in ALLOWED_HOSTS:                     # a redirect must not leave arXiv
        raise FigureError(f"The download was redirected to {final_host!r}, which is not allowed.")
    if resp.status_code != 200:
        raise FigureError(f"arXiv answered HTTP {resp.status_code} for the PDF.")
    declared = resp.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise FigureError(f"The PDF is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    data = bytearray()
    for chunk in resp.iter_content(chunk_size=65536):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise FigureError(f"The PDF is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if not bytes(data[:5]).startswith(b"%PDF"):
        raise FigureError("The downloaded file is not a PDF.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(data))
    return dest


def find_captions(page_text: str, max_chars: int = 300) -> List[Tuple[str, str]]:
    """Captions that START a line, such as 'Figure 2: Accuracy of ...'. In-text mentions are ignored."""
    found: List[Tuple[str, str]] = []
    matches = list(_CAPTION_START.finditer(page_text or ""))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        block = page_text[m.end():end]
        block = re.split(r"\n\s*\n", block, maxsplit=1)[0]          # a caption ends at a blank line
        text = re.sub(r"\s+", " ", block).strip()[:max_chars]
        if len(text) >= 8:
            found.append((f"Figure {m.group(1)}", text))
    return found


def extract_figures(pdf_path: Path, out_dir: Path, max_pages: int = 40, max_figures: int = 8,
                    scale: float = 1.4) -> List[FigureInfo]:
    try:
        from pypdf import PdfReader
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise FigureError("Figure extraction needs: pip install -r requirements-figures.txt") from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: List[FigureInfo] = []
    try:
        reader = PdfReader(str(pdf_path))
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        raise FigureError(f"The PDF could not be opened: {exc}") from exc
    try:
        rendered: dict = {}
        for index in range(min(len(reader.pages), max_pages)):
            if len(figures) >= max_figures:
                break
            try:
                text = reader.pages[index].extract_text() or ""
            except Exception as exc:                                  # a damaged page must not stop the others
                logger.warning("Skipping unreadable page %d: %s", index + 1, exc)
                continue
            for label, caption in find_captions(text):
                if len(figures) >= max_figures:
                    break
                if index not in rendered:
                    path = out_dir / f"page_{index + 1:03d}.png"
                    pdf[index].render(scale=scale).to_pil().save(path)
                    rendered[index] = path
                figures.append(FigureInfo(index + 1, label, caption, str(rendered[index])))
    except FigureError:
        raise
    except Exception as exc:
        raise FigureError(f"Rendering the PDF failed: {exc}") from exc
    finally:
        pdf.close()
    save_manifest(out_dir, figures)
    return figures


def save_manifest(directory: Path, figures: List[FigureInfo]) -> None:
    (directory / MANIFEST).write_text(json.dumps([asdict(f) for f in figures], indent=2), encoding="utf-8")


def load_manifest(directory: Path) -> List[FigureInfo]:
    path = Path(directory) / MANIFEST
    if not path.exists():
        return []
    try:
        return [FigureInfo(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    except (ValueError, TypeError):
        return []


def describe_figure(image_path: Path, caption: str, base_url: str, model: str, timeout: float = 300.0,
                    session: Optional[requests.Session] = None) -> str:
    """Ask a local vision model (through Ollama) to describe a page image. The answer is UNVERIFIED."""
    if not model:
        raise FigureError("No vision model is configured. Set FIGURE_VLM_MODEL in .env.")
    http = session or requests
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    prompt = ("This is a page from a research paper. Describe the figure whose caption is given, in two or three "
              "sentences. Say only what you can see. If you cannot tell, say so. Do not invent numbers.\n"
              f"Caption: {caption}")
    try:
        resp = http.post(f"{base_url.rstrip('/')}/api/chat", timeout=timeout, json={
            "model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}]})
    except requests.RequestException as exc:
        raise FigureError(f"Could not reach Ollama for the vision model: {exc}") from exc
    if resp.status_code != 200:
        raise FigureError(f"Ollama answered HTTP {resp.status_code}. Is '{model}' a vision model that you have pulled?")
    try:
        return str(resp.json()["message"]["content"]).strip()
    except (KeyError, ValueError) as exc:
        raise FigureError("Ollama returned an unexpected reply for the vision model.") from exc
