import pytest

from research_assistant.models import DocumentChunk, SearchResult
from research_assistant.retrieval import reciprocal_rank_fusion


def result(chunk_id, source, score):
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            source=source,
            text=source,
        ),
        score=score,
    )


def test_rrf_combines_results_from_multiple_retrievers():
    dense_results = [
        result("python:0", "python.md", 0.90),
        result("rag:0", "rag.md", 0.80),
    ]

    bm25_results = [
        result("rag:0", "rag.md", 5.00),
        result("database:0", "database.md", 4.00),
    ]

    fused = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )

    assert len(fused) == 3
    assert fused[0].chunk.source == "rag.md"


def test_rrf_preserves_unique_results():
    dense_results = [
        result("python:0", "python.md", 0.90),
    ]

    bm25_results = [
        result("database:0", "database.md", 5.00),
    ]

    fused = reciprocal_rank_fusion(
        [dense_results, bm25_results]
    )

    assert len(fused) == 2
    assert {
        item.chunk.source
        for item in fused
    } == {
        "python.md",
        "database.md",
    }


def test_rrf_respects_top_k():
    results_a = [
        result("a:0", "a.md", 0.90),
        result("b:0", "b.md", 0.80),
        result("c:0", "c.md", 0.70),
    ]

    results_b = [
        result("c:0", "c.md", 0.90),
        result("b:0", "b.md", 0.80),
        result("d:0", "d.md", 0.70),
    ]

    fused = reciprocal_rank_fusion(
        [results_a, results_b],
        top_k=2,
    )

    assert len(fused) == 2


def test_rrf_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        reciprocal_rank_fusion(
            [[]],
            top_k=0,
        )


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be at least 1"):
        reciprocal_rank_fusion(
            [[]],
            k=0,
        )
