import json
from pathlib import Path

import numpy as np

from .models import DocumentChunk


class IndexStore:
    """Persist chunks and their embeddings as a small local research index."""

    def __init__(self, directory: str | Path = "data/index"):
        self.directory = Path(directory)
        self.metadata_path = self.directory / "metadata.json"
        self.embeddings_path = self.directory / "embeddings.npy"

    def save(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(
                [{"chunk_id": c.chunk_id, "source": c.source, "text": c.text} for c in chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        np.save(self.embeddings_path, embeddings)

    def load(self) -> tuple[list[DocumentChunk], np.ndarray]:
        if not self.metadata_path.exists() or not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"No index found at {self.directory}. Run 'research-assistant index <directory>' first."
            )
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in raw]
        embeddings = np.load(self.embeddings_path)
        if len(chunks) != len(embeddings):
            raise ValueError("Index metadata and embeddings are out of sync")
        return chunks, embeddings
