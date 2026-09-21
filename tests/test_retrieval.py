import numpy as np
import pytest

from research_assistant.models import DocumentChunk
from research_assistant.retrieval import SemanticRetriever
from research_assistant.vector_store import QdrantVectorStore


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


def test_semantic_retriever_uses_configured_model(monkeypatch):
    model_names = []

    class ConfiguredEmbeddingModel:
        def __init__(self, model_name):
            model_names.append(model_name)

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        ConfiguredEmbeddingModel,
    )

    SemanticRetriever(
        model_name="test-embedding-model",
    )

    assert model_names == ["test-embedding-model"]


def test_semantic_retriever_filters_results_by_metadata(monkeypatch):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming",
            metadata={"topic": "python"},
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="Database systems",
            metadata={"topic": "database"},
        ),
    ]

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            if len(texts) == 2:
                return np.array([
                    [1.0, 0.0],
                    [0.0, 1.0],
                ])
            return np.array([[1.0, 0.0]])

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    retriever = SemanticRetriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=2,
        metadata_filter={"topic": "python"},
    )

    assert [result.chunk.chunk_id for result in results] == ["python:0"]


def test_semantic_retriever_returns_empty_for_non_matching_metadata_filter(
    monkeypatch,
):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming",
            metadata={"topic": "python"},
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="Database systems",
            metadata={"topic": "database"},
        ),
    ]

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            if len(texts) == 2:
                return np.array([
                    [1.0, 0.0],
                    [0.0, 1.0],
                ])
            return np.array([[1.0, 0.0]])

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    retriever = SemanticRetriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=2,
        metadata_filter={"topic": "rust"},
    )

    assert results == []


def test_semantic_retriever_requires_all_metadata_filters(monkeypatch):
    chunks = [
        DocumentChunk(
            chunk_id="python-tr:0",
            source="python-tr.md",
            text="Python programming",
            metadata={"topic": "python", "language": "tr"},
        ),
        DocumentChunk(
            chunk_id="python-en:0",
            source="python-en.md",
            text="Python programming",
            metadata={"topic": "python", "language": "en"},
        ),
        DocumentChunk(
            chunk_id="database-tr:0",
            source="database-tr.md",
            text="Database systems",
            metadata={"topic": "database", "language": "tr"},
        ),
    ]

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            if len(texts) == 3:
                return np.array([
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.8, 0.2],
                ])
            return np.array([[1.0, 0.0]])

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    retriever = SemanticRetriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=3,
        metadata_filter={
            "topic": "python",
            "language": "tr",
        },
    )

    assert [result.chunk.chunk_id for result in results] == ["python-tr:0"]


def test_semantic_retriever_can_use_vector_store(monkeypatch):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming",
        ),
    ]

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class FakeVectorStore:
        def __init__(self):
            self.chunks = None
            self.embeddings = None
            self.query_embedding = None
            self.top_k = None
            self.metadata_filter = None

        def upsert(self, chunks, embeddings):
            self.chunks = chunks
            self.embeddings = embeddings

        def search(
            self,
            query_embedding,
            top_k,
            metadata_filter=None,
        ):
            self.query_embedding = query_embedding
            self.top_k = top_k
            self.metadata_filter = metadata_filter

            return [(self.chunks[0], 0.95)]

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    vector_store = FakeVectorStore()
    retriever = SemanticRetriever(
        vector_store=vector_store,
    )

    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=3,
        metadata_filter={"topic": "python"},
    )

    assert results[0].chunk == chunks[0]
    assert results[0].score == 0.95
    assert vector_store.chunks == chunks
    assert np.array_equal(
        vector_store.embeddings,
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    assert np.array_equal(
        vector_store.query_embedding,
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert vector_store.top_k == 3
    assert vector_store.metadata_filter == {"topic": "python"}


def test_semantic_retriever_can_use_qdrant_vector_store(tmp_path, monkeypatch):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming",
            metadata={"topic": "python"},
        ),
        DocumentChunk(
            chunk_id="java:0",
            source="java.md",
            text="Java programming",
            metadata={"topic": "java"},
        ),
    ]

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            if len(texts) == 1:
                return np.array([[1.0, 0.0]], dtype=np.float32)

            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")
    retriever = SemanticRetriever(vector_store=vector_store)

    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=1,
        metadata_filter={"topic": "python"},
    )

    assert len(results) == 1
    assert results[0].chunk == chunks[0]
    assert results[0].score > 0.9

    vector_store.close()


def test_semantic_retriever_can_search_qdrant_without_loaded_embeddings(
    tmp_path,
    monkeypatch,
):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming",
        ),
        DocumentChunk(
            chunk_id="java:0",
            source="java.md",
            text="Java programming",
        ),
    ]

    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    vector_store.upsert(
        chunks,
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )

    retriever = SemanticRetriever(vector_store=vector_store)
    retriever.chunks = chunks
    retriever.embeddings = None

    results = retriever.search(
        "Python",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk == chunks[0]

    vector_store.close()


def test_semantic_retriever_does_not_keep_embeddings_with_vector_store(
    tmp_path,
    monkeypatch,
):
    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    chunks = [
        DocumentChunk(
            chunk_id="1",
            source="a.md",
            text="Python",
        ),
        DocumentChunk(
            chunk_id="2",
            source="b.md",
            text="Java",
        ),
    ]

    retriever = SemanticRetriever(vector_store=vector_store)
    retriever.fit(chunks)

    assert retriever.embeddings is None

    vector_store.close()
