from research_assistant.models import DocumentChunk, SearchResult
from research_assistant.pipeline import ResearchPipeline


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
    ]

    dense_results = [
        SearchResult(
            chunk=chunks[1],
            score=0.90,
        ),
        SearchResult(
            chunk=chunks[0],
            score=0.80,
        ),
        SearchResult(
            chunk=chunks[2],
            score=0.70,
        ),
    ]

    bm25_results = [
        SearchResult(
            chunk=chunks[0],
            score=5.00,
        ),
        SearchResult(
            chunk=chunks[2],
            score=3.00,
        ),
        SearchResult(
            chunk=chunks[1],
            score=1.00,
        ),
    ]

    class FakeSemanticRetriever:
        def __init__(self, *args, **kwargs):
            self.embeddings = None

        def search(self, question, top_k=8):
            return dense_results

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, question, top_k=8):
            return bm25_results

    class FakeReranker:
        def __init__(self, *args, **kwargs):
            pass

        def rerank(self, question, results, top_k=5):
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
        top_k=2,
    )

    assert "python.md" in answer
    assert answer.index("python.md") < answer.index("database.md")