import numpy as np
import re
from sentence_transformers import CrossEncoder, SentenceTransformer

from .models import DocumentChunk, SearchResult

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 8,
    k: int = 60,
) -> list[SearchResult]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if k < 1:
        raise ValueError("k must be at least 1")

    fused_scores: dict[str, float] = {}
    chunks: dict[str, DocumentChunk] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id

            chunks[chunk_id] = result.chunk
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (k + rank)
            )

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=score,
        )
        for chunk_id, score in ranked[:top_k]
    ]

class BM25Retriever:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.tokenized_chunks: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avg_doc_length = 0.0

    def fit(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty document collection")

        self.chunks = chunks
        self.tokenized_chunks = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        document_count = len(self.tokenized_chunks)
        total_length = sum(
            len(tokens) for tokens in self.tokenized_chunks
        )
        self.avg_doc_length = total_length / document_count

        document_frequency: dict[str, int] = {}

        for tokens in self.tokenized_chunks:
            for term in set(tokens):
                document_frequency[term] = (
                    document_frequency.get(term, 0) + 1
                )

        self.idf = {
            term: np.log(
                1
                + (document_count - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        top_k: int = 8,
    ) -> list[SearchResult]:
        if not self.tokenized_chunks:
            raise RuntimeError("Retriever is not fitted")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_terms = self._tokenize(query)

        scores = [
            self._score(query_terms, tokens)
            for tokens in self.tokenized_chunks
        ]

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                self.chunks[i],
                float(scores[i]),
            )
            for i in indices
        ]

    def _score(
        self,
        query_terms: list[str],
        document_terms: list[str],
    ) -> float:
        if not document_terms:
            return 0.0

        document_length = len(document_terms)
        term_frequencies: dict[str, int] = {}

        for term in document_terms:
            term_frequencies[term] = (
                term_frequencies.get(term, 0) + 1
            )

        score = 0.0

        for term in query_terms:
            if term not in self.idf:
                continue

            frequency = term_frequencies.get(term, 0)

            if frequency == 0:
                continue

            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1
                - self.b
                + self.b * document_length / self.avg_doc_length
            )

            score += self.idf[term] * numerator / denominator

        return score

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())


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
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-dimensional array")

        self.embeddings = embeddings

    def load(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-dimensional array")

        expected_dimension = self.model.get_embedding_dimension()
        if embeddings.shape[1] != expected_dimension:
            raise ValueError(
                f"embeddings must have dimension {expected_dimension}"
            )

        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Retriever is not fitted")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embeddings = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        if query_embeddings.ndim != 2 or query_embeddings.shape[0] != 1:
            raise ValueError(
                "query embedding must contain exactly one vector"
            )

        q = query_embeddings[0]

        if q.ndim != 1:
            raise ValueError(
                "query embedding must be a 1-dimensional vector"
            )

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
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
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
