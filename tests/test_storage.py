import numpy as np

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
