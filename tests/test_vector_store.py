import numpy as np
import pytest

from research_assistant.models import DocumentChunk
from research_assistant.vector_store import QdrantVectorStore


def test_qdrant_vector_store_upsert_and_search(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python is a programming language.",
            metadata={"topic": "python"},
        ),
        DocumentChunk(
            chunk_id="java:0",
            source="java.md",
            text="Java is a programming language.",
            metadata={"topic": "java"},
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = QdrantVectorStore(tmp_path / "qdrant")

    store.upsert(chunks, embeddings)

    results = store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0] == chunks[0]
    assert results[0][1] > 0.99


def test_qdrant_vector_store_filters_by_metadata(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python retrieval.",
            metadata={"topic": "python"},
        ),
        DocumentChunk(
            chunk_id="java:0",
            source="java.md",
            text="Java retrieval.",
            metadata={"topic": "java"},
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
        ],
        dtype=np.float32,
    )

    store = QdrantVectorStore(tmp_path / "qdrant")

    store.upsert(chunks, embeddings)

    results = store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=5,
        metadata_filter={"topic": "java"},
    )

    assert len(results) == 1
    assert results[0][0] == chunks[1]


def test_qdrant_vector_store_rejects_mismatched_embedding_count(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="a:0",
            source="a.md",
            text="first",
        ),
        DocumentChunk(
            chunk_id="b:0",
            source="b.md",
            text="second",
        ),
    ]

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    store = QdrantVectorStore(tmp_path / "qdrant")

    with pytest.raises(
        ValueError,
        match="chunks and embeddings must have the same length",
    ):
        store.upsert(chunks, embeddings)


def test_qdrant_vector_store_rejects_one_dimensional_embeddings(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="a:0",
            source="a.md",
            text="first",
        )
    ]

    embeddings = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    store = QdrantVectorStore(tmp_path / "qdrant")

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2-dimensional array",
    ):
        store.upsert(chunks, embeddings)


def test_qdrant_vector_store_rejects_invalid_query_embedding(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant")

    with pytest.raises(
        ValueError,
        match="query_embedding must be a 1-dimensional array",
    ):
        store.search(
            np.array([[1.0, 0.0]], dtype=np.float32),
            top_k=1,
        )


def test_qdrant_vector_store_rejects_invalid_top_k(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant")

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        store.search(
            np.array([1.0, 0.0], dtype=np.float32),
            top_k=0,
        )


def test_qdrant_vector_store_loads_chunks(tmp_path):
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

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store = QdrantVectorStore(tmp_path / "qdrant")
    store.upsert(chunks, embeddings)

    loaded_chunks = store.load_chunks()

    assert len(loaded_chunks) == 2
    assert {chunk.chunk_id for chunk in loaded_chunks} == {
        "python:0",
        "java:0",
    }
    assert loaded_chunks[0].metadata or loaded_chunks[1].metadata
    assert {
        chunk.metadata["topic"]
        for chunk in loaded_chunks
    } == {"python", "java"}


def test_qdrant_vector_store_can_close(tmp_path):
    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    vector_store.close()


def test_qdrant_vector_store_replaces_previous_chunks(tmp_path):
    store = QdrantVectorStore(tmp_path / "qdrant")

    first_chunks = [
        DocumentChunk(
            chunk_id="old-1",
            source="old.md",
            text="Old content",
        ),
        DocumentChunk(
            chunk_id="old-2",
            source="old.md",
            text="Stale content",
        ),
    ]

    first_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.upsert(first_chunks, first_embeddings)

    second_chunks = [
        DocumentChunk(
            chunk_id="new-1",
            source="new.md",
            text="New content",
        ),
    ]

    second_embeddings = np.array(
        [[1.0, 1.0]],
        dtype=np.float32,
    )

    store.upsert(second_chunks, second_embeddings)

    chunks = store.load_chunks()

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "new-1"

    store.close()