import numpy as np
import pytest

from research_assistant.models import DocumentChunk
from research_assistant.retrieval import SemanticRetriever


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        embeddings = []

        for text in texts:
            text = text.lower()

            if "python" in text:
                vector = [1.0, 0.0]
            elif "database" in text:
                vector = [0.0, 1.0]
            else:
                vector = [0.5, 0.5]

            embeddings.append(vector)

        return np.array(embeddings, dtype=np.float32)


def create_retriever(monkeypatch):
    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda *args, **kwargs: FakeEmbeddingModel(),
    )
    return SemanticRetriever()


def test_semantic_retriever_returns_most_relevant_chunk(monkeypatch):
    retriever = create_retriever(monkeypatch)

    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python is widely used for machine learning.",
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="PostgreSQL is a relational database.",
        ),
    ]

    retriever.fit(chunks)

    assert retriever.embeddings.shape == (2, 2)

    results = retriever.search(
        "Python programming",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.source == "python.md"


def test_semantic_retriever_respects_top_k(monkeypatch):
    retriever = create_retriever(monkeypatch)

    chunks = [
        DocumentChunk(
            "python:0",
            "python.md",
            "Python programming",
        ),
        DocumentChunk(
            "database:0",
            "database.md",
            "Database systems",
        ),
    ]

    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.source == "python.md"


def test_semantic_retriever_rejects_invalid_top_k(monkeypatch):
    retriever = create_retriever(monkeypatch)

    chunks = [
        DocumentChunk(
            "python:0",
            "python.md",
            "Python programming",
        ),
    ]

    retriever.fit(chunks)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search(
            "Python",
            top_k=0,
        )


def test_semantic_retriever_load_rejects_non_2d_embeddings(monkeypatch):
    retriever = create_retriever(monkeypatch)

    chunks = [
        DocumentChunk(
            "python:0",
            "python.md",
            "Python programming",
        ),
        DocumentChunk(
            "database:0",
            "database.md",
            "Database systems",
        ),
    ]

    embeddings = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2-dimensional array",
    ):
        retriever.load(chunks, embeddings)


def test_semantic_retriever_fit_rejects_non_2d_embeddings(monkeypatch):
    class InvalidEmbeddingModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            return np.array([1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda *args, **kwargs: InvalidEmbeddingModel(),
    )

    retriever = SemanticRetriever()

    chunks = [
        DocumentChunk(
            "python:0",
            "python.md",
            "Python programming",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2-dimensional array",
    ):
        retriever.fit(chunks)


def test_semantic_retriever_search_rejects_invalid_query_embedding(
    monkeypatch,
):
    class InvalidQueryEmbeddingModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            if len(texts) == 1:
                return np.array([1.0, 0.0], dtype=np.float32)

            return np.array(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            )

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda *args, **kwargs: InvalidQueryEmbeddingModel(),
    )

    retriever = SemanticRetriever()

    chunks = [
        DocumentChunk(
            "python:0",
            "python.md",
            "Python programming",
        ),
        DocumentChunk(
            "database:0",
            "database.md",
            "Database systems",
        ),
    ]

    retriever.fit(chunks)

    with pytest.raises(
        ValueError,
        match="query embedding must contain exactly one vector",
    ):
        retriever.search("Python")


def test_load_rejects_wrong_embedding_dimension():
    retriever = SemanticRetriever()
    chunks = [
        DocumentChunk(
            chunk_id="1",
            source="test.md",
            text="retrieval",
        )
    ]
    embeddings = np.zeros((1, 128))

    with pytest.raises(
        ValueError,
        match="embeddings must have dimension 384",
    ):
        retriever.load(chunks, embeddings)
