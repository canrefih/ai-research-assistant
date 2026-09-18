import numpy as np
import pytest

from research_assistant.models import DocumentChunk, SearchResult
from research_assistant.retrieval import Reranker


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []

        for _, text in pairs:
            text = text.lower()

            if "python" in text:
                scores.append(0.95)
            elif "database" in text:
                scores.append(0.25)
            else:
                scores.append(0.10)

        return np.array(scores, dtype=np.float32)


def create_reranker(monkeypatch):
    monkeypatch.setattr(
        "research_assistant.retrieval.CrossEncoder",
        lambda *args, **kwargs: FakeCrossEncoder(),
    )
    return Reranker()


def test_reranker_prefers_relevant_chunk(monkeypatch):
    reranker = create_reranker(monkeypatch)

    results = [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="database:0",
                source="database.md",
                text="PostgreSQL is a relational database.",
            ),
            score=0.40,
        ),
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="python:0",
                source="python.md",
                text="Python is widely used for machine learning.",
            ),
            score=0.60,
        ),
    ]

    reranked = reranker.rerank(
        "Python programming",
        results,
        top_k=2,
    )

    assert len(reranked) == 2
    assert reranked[0].chunk.source == "python.md"
    assert reranked[0].score > reranked[1].score


def test_reranker_respects_top_k(monkeypatch):
    reranker = create_reranker(monkeypatch)

    results = [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="python:0",
                source="python.md",
                text="Python programming",
            ),
            score=0.60,
        ),
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="database:0",
                source="database.md",
                text="Database systems",
            ),
            score=0.40,
        ),
    ]

    reranked = reranker.rerank(
        "Python",
        results,
        top_k=1,
    )

    assert len(reranked) == 1
    assert reranked[0].chunk.source == "python.md"


def test_reranker_rejects_invalid_top_k(monkeypatch):
    reranker = create_reranker(monkeypatch)

    results = [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id="python:0",
                source="python.md",
                text="Python programming",
            ),
            score=0.60,
        ),
    ]

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        reranker.rerank(
            "Python",
            results,
            top_k=0,
        )