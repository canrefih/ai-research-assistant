from pathlib import Path
from .chunking import chunk_text
from .models import DocumentChunk

SUPPORTED = {".txt", ".md"}

def load_directory(directory: str | Path) -> list[DocumentChunk]:
    root = Path(directory)
    chunks: list[DocumentChunk] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            chunks.extend(chunk_text(path.read_text(encoding="utf-8"), str(path)))
    return chunks
