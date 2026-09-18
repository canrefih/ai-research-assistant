from dataclasses import dataclass

@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str

@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
