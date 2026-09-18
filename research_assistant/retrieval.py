import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from .models import DocumentChunk, SearchResult

class SemanticRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[DocumentChunk] = []
        self.embeddings = None

    def fit(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("No supported documents found")

        self.chunks = chunks
        self.embeddings = self.model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Retriever is not fitted")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self.embeddings @ query_embedding
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(self.chunks[i], float(scores[i]))
            for i in indices
        ]

class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        pairs = [(query, result.chunk.text) for result in results]
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
