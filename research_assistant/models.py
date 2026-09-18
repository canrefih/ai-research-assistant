from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float