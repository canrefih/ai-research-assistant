import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .models import DocumentChunk, SearchResult


class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None

    def fit(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty document collection")
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def load(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Retriever is not fitted")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        scores = self.embeddings @ q
        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in indices]


class Reranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(results, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            SearchResult(result.chunk, float(score))
            for result, score in ranked[:top_k]
        ]
