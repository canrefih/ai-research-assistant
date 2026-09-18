from pathlib import Path

from .chunking import chunk_text
from .models import DocumentChunk


SUPPORTED = {".txt", ".md"}


def load_directory(directory: str | Path) -> list[DocumentChunk]:
    root = Path(directory)
    chunks: list[DocumentChunk] = []
    seen_texts: set[str] = set()

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            file_chunks = chunk_text(
                path.read_text(encoding="utf-8"),
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