from research_assistant.models import DocumentChunk, SearchResult
from research_assistant.pipeline import ResearchPipeline

import pytest


def test_pipeline_uses_hybrid_retrieval(monkeypatch, tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming and machine learning.",
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="PostgreSQL relational database systems.",
        ),
        DocumentChunk(
            chunk_id="rag:0",
            source="rag.md",
            text="Retrieval augmented generation combines search and generation.",
        ),
        DocumentChunk(
            chunk_id="ai:0",
            source="ai.md",
            text="Artificial intelligence and neural networks.",
        ),
        DocumentChunk(
            chunk_id="search:0",
            source="search.md",
            text="Search engines retrieve relevant documents.",
        ),
        DocumentChunk(
            chunk_id="ml:0",
            source="ml.md",
            text="Machine learning models process data.",
        ),
    ]

    dense_results = [
        SearchResult(chunk=chunks[1], score=0.90),
        SearchResult(chunk=chunks[0], score=0.80),
        SearchResult(chunk=chunks[2], score=0.70),
        SearchResult(chunk=chunks[3], score=0.60),
        SearchResult(chunk=chunks[4], score=0.50),
        SearchResult(chunk=chunks[5], score=0.40),
    ]

    bm25_results = [
        SearchResult(chunk=chunks[0], score=5.00),
        SearchResult(chunk=chunks[2], score=3.00),
        SearchResult(chunk=chunks[1], score=1.00),
        SearchResult(chunk=chunks[3], score=0.90),
        SearchResult(chunk=chunks[4], score=0.80),
        SearchResult(chunk=chunks[5], score=0.70),
    ]

    reranker_top_k = []

    class FakeSemanticRetriever:
        def __init__(self, *args, **kwargs):
            self.embeddings = None

        def search(self, question, top_k=8):
            return dense_results[:top_k]

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, question, top_k=8):
            return bm25_results[:top_k]

    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, question, results, top_k=5):
            reranker_top_k.append(top_k)
            return results[:top_k]

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def answer(self, question, evidence):
            return evidence

    monkeypatch.setattr(
        "research_assistant.pipeline.SemanticRetriever",
        FakeSemanticRetriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.BM25Retriever",
        FakeBM25Retriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.Reranker",
        FakeReranker,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        FakeLLM,
    )

    pipeline = ResearchPipeline(
        use_reranker=True,
        index_dir=tmp_path,
    )

    answer = pipeline.ask(
        "Python programming",
        top_k=6,
    )

    assert reranker_top_k == [6]
    assert "python.md" in answer


def test_pipeline_rejects_empty_question(monkeypatch, tmp_path):
    class FakeSemanticRetriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "research_assistant.pipeline.SemanticRetriever",
        FakeSemanticRetriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.BM25Retriever",
        FakeBM25Retriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        FakeLLM,
    )

    pipeline = ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        pipeline.ask("   ")


def test_pipeline_rejects_invalid_top_k(monkeypatch, tmp_path):
    class FakeSemanticRetriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "research_assistant.pipeline.SemanticRetriever",
        FakeSemanticRetriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.BM25Retriever",
        FakeBM25Retriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        FakeLLM,
    )

    pipeline = ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        pipeline.ask(
            "Python",
            top_k=0,
        )
