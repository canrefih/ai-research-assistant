import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from .models import DocumentChunk, SearchResult

class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[DocumentChunk] = []
        self.embeddings = None

    def fit(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        if self.embeddings is None:
            raise RuntimeError("Retriever is not fitted")
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.embeddings @ q
        indices = np.argsort(scores)[::-1][:top_k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in indices]

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(results, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        return [SearchResult(r.chunk, float(score)) for r, score in ranked[:top_k]]
