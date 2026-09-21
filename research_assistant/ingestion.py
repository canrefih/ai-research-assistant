from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .chunking import chunk_text
from .models import DocumentChunk


SUPPORTED = {".txt", ".md", ".pdf", ".html", ".htm"}


def load_directory(directory: str | Path) -> list[DocumentChunk]:
    root = Path(directory)
    chunks: list[DocumentChunk] = []
    seen_texts: set[str] = set()

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            text = (
                _read_pdf(path)
                if path.suffix.lower() == ".pdf"
                else _read_html(path)
                if path.suffix.lower() in {".html", ".htm"}
                else path.read_text(encoding="utf-8")
            )

            file_chunks = chunk_text(
                text,
                str(path),
                metadata={
                    "source": path.name,
                    "file_type": path.suffix.lower().lstrip("."),
                },
            )

            for chunk in file_chunks:
                if chunk.text in seen_texts:
                    continue

                seen_texts.add(chunk.text)
                chunks.append(chunk)

    return chunks


def _read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_html(path: Path) -> str:
    return _parse_html(
        path.read_text(encoding="utf-8"),
    )


def _parse_html(html: str) -> str:
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    return soup.get_text(" ", strip=True)


def _fetch_html(url: str) -> str:
    response = requests.get(
        url,
        timeout=10,
    )
    response.raise_for_status()
    return response.text


def fetch_web_page(url: str) -> str:
    html = _fetch_html(url)
    return _parse_html(html)


def load_url(url: str) -> list[DocumentChunk]:
    text = fetch_web_page(url)

    return chunk_text(
        text,
        url,
        metadata={
            "source": url,
            "file_type": "html",
        },
    )