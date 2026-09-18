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

        def search(self, question, top_k=8, metadata_filter=None):
            return dense_results[:top_k]


    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, question, top_k=8, metadata_filter=None):
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


def test_pipeline_loads_index_into_both_retrievers(monkeypatch, tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="1",
            source="test.md",
            text="retrieval",
        ),
        DocumentChunk(
            chunk_id="2",
            source="test.md",
            text="generation",
        ),
    ]

    embeddings = object()

    class FakeStore:
        def __init__(self, directory):
            pass

        def load(self):
            return chunks, embeddings

    semantic_loads = []
    bm25_fits = []

    class FakeSemanticRetriever:
        def __init__(self, *args, **kwargs):
            pass

        def load(self, loaded_chunks, loaded_embeddings):
            semantic_loads.append(
                (loaded_chunks, loaded_embeddings)
            )

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, loaded_chunks):
            bm25_fits.append(loaded_chunks)

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "research_assistant.pipeline.IndexStore",
        FakeStore,
    )
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

    count = pipeline.load_index()

    assert count == 2
    assert semantic_loads == [(chunks, embeddings)]
    assert bm25_fits == [chunks]


def test_pipeline_load_index_propagates_missing_index_error(
    monkeypatch,
    tmp_path,
):
    class FakeStore:
        def __init__(self, directory):
            pass

        def load(self):
            raise FileNotFoundError("No index found")

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
        "research_assistant.pipeline.IndexStore",
        FakeStore,
    )
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
        FileNotFoundError,
        match="No index found",
    ):
        pipeline.load_index()


def test_pipeline_passes_embedding_model_to_retriever(
    monkeypatch,
    tmp_path,
):
    model_names = []

    class FakeSemanticRetriever:
        def __init__(self, model_name):
            model_names.append(model_name)

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

    ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path,
        embedding_model="test-embedding-model",
    )

    assert model_names == ["test-embedding-model"]


def test_pipeline_passes_reranker_model_to_reranker(
    monkeypatch,
    tmp_path,
):
    model_names = []

    class FakeSemanticRetriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeReranker:
        def __init__(self, model_name):
            model_names.append(model_name)

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
        "research_assistant.pipeline.Reranker",
        FakeReranker,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        FakeLLM,
    )

    ResearchPipeline(
        use_reranker=True,
        index_dir=tmp_path,
        reranker_model="test-reranker-model",
    )

    assert model_names == ["test-reranker-model"]


def test_pipeline_passes_metadata_filter_to_retrievers(monkeypatch):
    pipeline = ResearchPipeline(use_reranker=False)

    captured = {}

    def fake_semantic_search(query, top_k=8, metadata_filter=None):
        captured["semantic"] = metadata_filter
        return []

    def fake_bm25_search(query, top_k=8, metadata_filter=None):
        captured["bm25"] = metadata_filter
        return []

    monkeypatch.setattr(
        pipeline.retriever,
        "search",
        fake_semantic_search,
    )
    monkeypatch.setattr(
        pipeline.bm25_retriever,
        "search",
        fake_bm25_search,
    )

    class FakeLLM:
        def answer(self, question, evidence):
            return "answer"

    pipeline.llm = FakeLLM()

    pipeline.ask(
        "Python nedir?",
        top_k=2,
        metadata_filter={"topic": "python"},
    )

    assert captured["semantic"] == {"topic": "python"}
    assert captured["bm25"] == {"topic": "python"}


def test_pipeline_accepts_custom_index_store():
    class FakeIndexStore:
        def exists(self):
            return False

        def save(self, chunks, embeddings):
            pass

        def load(self):
            return [], np.empty((0, 2))

    store = FakeIndexStore()
    pipeline = ResearchPipeline(
        use_reranker=False,
        store=store,
    )

    assert pipeline.store is store