import numpy as np
import pytest

from research_assistant.models import DocumentChunk
from research_assistant.storage import IndexStore


def test_index_store_round_trip(tmp_path):
    chunks = [DocumentChunk("a:0", "a.md", "hello world")]
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    store = IndexStore(tmp_path / "index")
    store.save(chunks, embeddings)

    loaded_chunks, loaded_embeddings = store.load()

    assert loaded_chunks == chunks
    assert np.array_equal(loaded_embeddings, embeddings)


def test_index_store_rejects_mismatched_embedding_shape(tmp_path):
    chunks = [
        DocumentChunk("a:0", "a.md", "hello world"),
        DocumentChunk("b:0", "b.md", "goodbye world"),
    ]
    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    store = IndexStore(tmp_path / "index")

    with pytest.raises(
        ValueError,
        match="chunks and embeddings must have the same length",
    ):
        store.save(chunks, embeddings)


def test_index_store_rejects_one_dimensional_embeddings(tmp_path):
    chunks = [
        DocumentChunk("a:0", "a.md", "hello world"),
        DocumentChunk("b:0", "b.md", "goodbye world"),
    ]
    embeddings = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    store = IndexStore(tmp_path / "index")
    store.directory.mkdir(parents=True, exist_ok=True)

    store.metadata_path.write_text(
        """
        [
            {"chunk_id": "a:0", "source": "a.md", "text": "hello world"},
            {"chunk_id": "b:0", "source": "b.md", "text": "goodbye world"}
        ]
        """,
        encoding="utf-8",
    )
    np.save(store.embeddings_path, embeddings)

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2-dimensional array",
    ):
        store.load()


def test_index_store_rejects_invalid_metadata_json(tmp_path):
    store = IndexStore(tmp_path / "index")
    store.directory.mkdir(parents=True, exist_ok=True)

    store.metadata_path.write_text(
        '{"chunk_id": "a:0", "source": "a.md",',
        encoding="utf-8",
    )
    np.save(
        store.embeddings_path,
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match="Invalid index metadata",
    ):
        store.load()


def test_index_store_rejects_missing_index(tmp_path):
    store = IndexStore(tmp_path / "index")

    with pytest.raises(
        FileNotFoundError,
        match="No index found",
    ):
        store.load()


def test_index_store_rejects_one_dimensional_embeddings_on_save(tmp_path):
    store = IndexStore(tmp_path)
    chunks = [
        DocumentChunk(
            chunk_id="1",
            source="test.md",
            text="retrieval",
        ),
        DocumentChunk(
            chunk_id="2",
            source="test.md",
            text="search",
        ),
        DocumentChunk(
            chunk_id="3",
            source="test.md",
            text="embedding",
        ),
    ]
    embeddings = np.zeros(3)

    with pytest.raises(
        ValueError,
        match="embeddings must be a 2-dimensional array",
    ):
        store.save(chunks, embeddings)


def test_index_store_preserves_chunk_metadata(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="test:0",
            source="test.md",
            text="Example text",
            metadata={"topic": "python"},
        )
    ]
    embeddings = np.zeros((1, 3))

    store = IndexStore(tmp_path)

    store.save(chunks, embeddings)
    loaded_chunks, loaded_embeddings = store.load()

    assert loaded_chunks == chunks
    assert np.array_equal(loaded_embeddings, embeddings)