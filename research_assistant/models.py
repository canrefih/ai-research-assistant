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


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class ResearchSource:
    source: str
    title: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class ResearchReport:
    question: str
    answer: str
    sources: list[ResearchSource]
