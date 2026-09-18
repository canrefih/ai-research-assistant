import pytest

from research_assistant.models import DocumentChunk
from research_assistant.retrieval import BM25Retriever


def create_chunks():
    return [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python is widely used for machine learning.",
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="PostgreSQL is a relational database system.",
        ),
        DocumentChunk(
            chunk_id="rag:0",
            source="rag.md",
            text="RAG combines retrieval with language generation.",
        ),
    ]


def test_bm25_returns_relevant_chunk():
    retriever = BM25Retriever()
    retriever.fit(create_chunks())

    results = retriever.search(
        "Python programming",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.source == "python.md"


def test_bm25_respects_top_k():
    retriever = BM25Retriever()
    retriever.fit(create_chunks())

    results = retriever.search(
        "database",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.source == "database.md"


def test_bm25_rejects_invalid_top_k():
    retriever = BM25Retriever()
    retriever.fit(create_chunks())

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.search(
            "Python",
            top_k=0,
        )


def test_bm25_rejects_empty_query():
    retriever = BM25Retriever()
    retriever.fit(create_chunks())

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.search(
            "   ",
            top_k=1,
        )


def test_bm25_rejects_search_before_fit():
    retriever = BM25Retriever()

    with pytest.raises(RuntimeError, match="Retriever is not fitted"):
        retriever.search(
            "Python",
            top_k=1,
        )


def test_bm25_handles_punctuation():
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python, is widely used for machine learning.",
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="PostgreSQL is a relational database system.",
        ),
    ]

    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=1,
    )

    assert results[0].chunk.source == "python.md"


def test_bm25_normalizes_query_punctuation():
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python is widely used for machine learning.",
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="PostgreSQL is a relational database system.",
        ),
    ]

    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python, programming!",
        top_k=1,
    )

    assert results[0].chunk.source == "python.md"


def test_bm25_rejects_empty_collection():
    retriever = BM25Retriever()

    with pytest.raises(
        ValueError,
        match="Cannot index an empty document collection",
    ):
        retriever.fit([])


def test_bm25_retriever_filters_results_by_metadata():
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
            text="Python database systems",
            metadata={"topic": "database"},
        ),
    ]

    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=2,
        metadata_filter={"topic": "python"},
    )

    assert [result.chunk.chunk_id for result in results] == ["python:0"]


def test_bm25_retriever_returns_empty_for_non_matching_metadata_filter():
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

    retriever = BM25Retriever()
    retriever.fit(chunks)

    results = retriever.search(
        "Python",
        top_k=2,
        metadata_filter={"topic": "rust"},
    )

    assert results == []


def test_bm25_retriever_requires_all_metadata_filters():
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

    retriever = BM25Retriever()
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