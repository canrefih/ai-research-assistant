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