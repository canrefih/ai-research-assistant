from pathlib import Path

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
    soup = BeautifulSoup(
        path.read_text(encoding="utf-8"),
        "html.parser",
    )

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    return soup.get_text(" ", strip=True)