from research_assistant.models import DocumentChunk, SearchResult, ResearchReport
from research_assistant.pipeline import ResearchPipeline
from research_assistant.vector_store import QdrantVectorStore
from research_assistant.models import WebSearchResult
from research_assistant.source_verification import SourceVerificationResult

import pytest
import numpy as np

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
        def __init__(self, model_name, vector_store=None):
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


def test_pipeline_passes_vector_store_to_retriever(monkeypatch, tmp_path):
    vector_stores = []

    class FakeSemanticRetriever:
        def __init__(self, model_name, vector_store=None):
            vector_stores.append(vector_store)

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

    vector_store = object()

    ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path,
        vector_store=vector_store,
    )

    assert vector_stores == [vector_store]


def test_pipeline_uses_app_config(monkeypatch, tmp_path):
    from research_assistant.config import AppConfig

    captured = {}

    class FakeSemanticRetriever:
        def __init__(self, model_name, vector_store=None):
            captured["embedding_model"] = model_name

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

    class FakeReranker:
        def __init__(self, model_name):
            captured["reranker_model"] = model_name

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

    config = AppConfig(
        index_dir=tmp_path,
        embedding_model="config-embedding-model",
        reranker_model="config-reranker-model",
        use_reranker=True,
    )

    ResearchPipeline(config=config)

    assert captured["embedding_model"] == "config-embedding-model"
    assert captured["reranker_model"] == "config-reranker-model"


def test_pipeline_indexes_into_qdrant(monkeypatch, tmp_path):
    class FakeModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            embeddings = []

            for text in texts:
                if "Python" in text:
                    embeddings.append([1.0, 0.0])
                else:
                    embeddings.append([0.0, 1.0])

            return np.array(embeddings, dtype=np.float32)

        def get_embedding_dimension(self):
            return 2

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    (documents_dir / "python.md").write_text(
        "# Python\n\nPython programming.",
        encoding="utf-8",
    )
    (documents_dir / "java.md").write_text(
        "# Java\n\nJava programming.",
        encoding="utf-8",
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    pipeline = ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path / "index",
        vector_store=vector_store,
    )

    count = pipeline.index(documents_dir)

    assert count == 2

    results = vector_store.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0].text == "# Python Python programming."
    assert results[0][1] > 0.9


def test_pipeline_loads_qdrant_index_without_loading_embeddings(
    monkeypatch,
    tmp_path,
):
    loaded = False

    class FakeSemanticRetriever:
        def __init__(self, model_name, vector_store=None):
            self.vector_store = vector_store

        def load(self, chunks, embeddings):
            nonlocal loaded
            loaded = True

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, chunks):
            pass

    class FakeStore:
        def load(self):
            raise AssertionError(
                "IndexStore should not be used when a vector store is configured"
            )

    class FakeVectorStore:
        def load_chunks(self):
            return []

    monkeypatch.setattr(
        "research_assistant.pipeline.SemanticRetriever",
        FakeSemanticRetriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.BM25Retriever",
        FakeBM25Retriever,
    )

    pipeline = ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path,
        store=FakeStore(),
        vector_store=FakeVectorStore(),
    )

    pipeline.load_index()

    assert not loaded


def test_pipeline_loads_and_searches_qdrant_index(
    monkeypatch,
    tmp_path,
):
    class FakeEmbeddingModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            embeddings = []

            for text in texts:
                if "Python" in text:
                    embeddings.append([1.0, 0.0])
                else:
                    embeddings.append([0.0, 1.0])

            return np.array(embeddings, dtype=np.float32)

        def get_embedding_dimension(self):
            return 2

    class FakeLLM:
        def answer(self, question, results):
            return "Python sonucu"

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(),
    )

    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        lambda: FakeLLM(),
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    chunks = [
        DocumentChunk(
            chunk_id="python:0",
            source="python.md",
            text="Python programming",
        ),
        DocumentChunk(
            chunk_id="database:0",
            source="database.md",
            text="Database systems",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    vector_store.upsert(chunks, embeddings)

    pipeline = ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path / "unused",
        vector_store=vector_store,
    )

    loaded = pipeline.load_index()

    assert loaded == 2

    results = pipeline.retriever.search(
        "Python",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.source == "python.md"


def test_pipeline_does_not_save_local_embeddings_when_using_vector_store(
    monkeypatch,
    tmp_path,
):
    class FakeStore:
        def save(self, chunks, embeddings):
            raise AssertionError(
                "IndexStore should not save embeddings when vector store is configured"
            )

    class FakeVectorStore:
        def upsert(self, chunks, embeddings):
            self.chunks = chunks
            self.embeddings = embeddings

    class FakeSemanticRetriever:
        def __init__(self, model_name, vector_store=None):
            self.vector_store = vector_store
            self.embeddings = np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            )

        def fit(self, chunks):
            self.chunks = chunks
            self.vector_store.upsert(
                chunks,
                self.embeddings,
            )

    class FakeBM25Retriever:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, chunks):
            pass

    monkeypatch.setattr(
        "research_assistant.pipeline.SemanticRetriever",
        FakeSemanticRetriever,
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.BM25Retriever",
        FakeBM25Retriever,
    )

    vector_store = FakeVectorStore()

    pipeline = ResearchPipeline(
        use_reranker=False,
        index_dir=tmp_path,
        store=FakeStore(),
        vector_store=vector_store,
    )

    monkeypatch.setattr(
        "research_assistant.pipeline.load_directory",
        lambda directory: [
            DocumentChunk(
                chunk_id="python:0",
                source="python.md",
                text="Python programming",
            )
        ],
    )

    assert pipeline.index(tmp_path) == 1


def test_pipeline_ask_uses_qdrant_and_bm25_hybrid_retrieval(
    monkeypatch,
    tmp_path,
):
    class FakeEmbeddingModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            embeddings = []

            for text in texts:
                if "Python" in text:
                    embeddings.append([1.0, 0.0])
                else:
                    embeddings.append([0.0, 1.0])

            return np.array(embeddings, dtype=np.float32)

        def get_embedding_dimension(self):
            return 2

    captured = {}

    class FakeLLM:
        def answer(self, question, evidence):
            captured["question"] = question
            captured["evidence"] = evidence
            return "Python sonucu"

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        lambda: FakeLLM(),
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    pipeline = ResearchPipeline(
        use_reranker=False,
        vector_store=vector_store,
    )

    monkeypatch.setattr(
        "research_assistant.pipeline.load_directory",
        lambda directory: [
            DocumentChunk(
                chunk_id="python:0",
                source="python.md",
                text="Python programming language",
            ),
            DocumentChunk(
                chunk_id="database:0",
                source="database.md",
                text="Database systems",
            ),
        ],
    )

    assert pipeline.index(tmp_path) == 2

    answer = pipeline.ask(
        "Python",
        top_k=1,
    )

    assert answer == "Python sonucu"
    assert captured["question"] == "Python"
    assert "[1] Source: python.md" in captured["evidence"]
    assert "Python programming language" in captured["evidence"]


def test_pipeline_ask_applies_metadata_filter_with_qdrant(
    monkeypatch,
    tmp_path,
):
    class FakeEmbeddingModel:
        def encode(
            self,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ):
            return np.array(
                [[1.0, 0.0] for _ in texts],
                dtype=np.float32,
            )

        def get_embedding_dimension(self):
            return 2

    captured = {}

    class FakeLLM:
        def answer(self, question, evidence):
            captured["evidence"] = evidence
            return "filtered result"

    monkeypatch.setattr(
        "research_assistant.retrieval.SentenceTransformer",
        lambda model_name: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        "research_assistant.pipeline.LLMClient",
        lambda: FakeLLM(),
    )

    vector_store = QdrantVectorStore(tmp_path / "qdrant")

    pipeline = ResearchPipeline(
        use_reranker=False,
        vector_store=vector_store,
    )

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

    monkeypatch.setattr(
        "research_assistant.pipeline.load_directory",
        lambda directory: chunks,
    )

    assert pipeline.index(tmp_path) == 2

    answer = pipeline.ask(
        "programming",
        top_k=2,
        metadata_filter={"topic": "python"},
    )

    assert answer == "filtered result"
    assert "[1] Source: python.md" in captured["evidence"]
    assert "Python programming" in captured["evidence"]
    assert "database.md" not in captured["evidence"]


def test_pipeline_can_close_vector_store():
    class FakeVectorStore:
        def __init__(self):
            self.closed = False

        def upsert(self, chunks, embeddings):
            pass

        def search(self, query_embedding, top_k, metadata_filter=None):
            return []

        def load_chunks(self):
            return []

        def close(self):
            self.closed = True

    vector_store = FakeVectorStore()

    pipeline = ResearchPipeline(
        use_reranker=False,
        vector_store=vector_store,
    )

    pipeline.close()

    assert vector_store.closed is True


def test_index_url(monkeypatch):
    from research_assistant.models import DocumentChunk

    chunks = [
        DocumentChunk(
            chunk_id="web-1",
            source="https://example.com",
            text="Web research content",
        )
    ]

    monkeypatch.setattr(
        "research_assistant.pipeline.load_url",
        lambda url: chunks,
    )

    class FakeRetriever:
        embeddings = [[1.0, 2.0]]

        def fit(self, chunks):
            assert chunks == [
                DocumentChunk(
                    chunk_id="web-1",
                    source="https://example.com",
                    text="Web research content",
                )
            ]

    class FakeBM25:
        def fit(self, chunks):
            assert chunks == [
                DocumentChunk(
                    chunk_id="web-1",
                    source="https://example.com",
                    text="Web research content",
                )
            ]

    class FakeStore:
        def save(self, chunks, embeddings):
            assert chunks == [
                DocumentChunk(
                    chunk_id="web-1",
                    source="https://example.com",
                    text="Web research content",
                )
            ]
            assert embeddings == [[1.0, 2.0]]

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25()
    pipeline.vector_store = None
    pipeline.store = FakeStore()

    assert pipeline.index_url("https://example.com") == 1


from research_assistant.models import WebSearchResult
from research_assistant.web_search import WebSearchProvider


class FakeWebSearchProvider(WebSearchProvider):
    def __init__(self):
        self.calls = []

    def search(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        return [
            WebSearchResult(
                title="Test result",
                url="https://example.com",
                snippet="Test snippet",
            )
        ]


def test_pipeline_search_web_uses_provider():
    provider = FakeWebSearchProvider()
    pipeline = ResearchPipeline(
        use_reranker=False,
        web_search_provider=provider,
    )

    results = pipeline.search_web(
        "retrieval augmented generation",
        top_k=3,
    )

    assert results == [
        WebSearchResult(
            title="Test result",
            url="https://example.com",
            snippet="Test snippet",
        )
    ]
    assert provider.calls == [
        ("retrieval augmented generation", 3)
    ]


def test_pipeline_search_web_requires_provider():
    pipeline = ResearchPipeline(use_reranker=False)

    with pytest.raises(ValueError, match="web search provider is not configured"):
        pipeline.search_web("test")


def test_crawl_url(monkeypatch):
    chunks = [
        DocumentChunk(
            chunk_id="web-1",
            source="https://example.com",
            text="Web research content",
        )
    ]

    class FakeCrawler:
        def __init__(self, max_pages):
            assert max_pages == 3

        def crawl_chunks(self, url):
            assert url == "https://example.com"
            return chunks

    monkeypatch.setattr(
        "research_assistant.crawler.WebCrawler",
        FakeCrawler,
    )

    class FakeRetriever:
        embeddings = [[1.0, 2.0]]

        def fit(self, chunks):
            assert chunks == [
                DocumentChunk(
                    chunk_id="web-1",
                    source="https://example.com",
                    text="Web research content",
                )
            ]

    class FakeBM25:
        def fit(self, chunks):
            assert chunks == [
                DocumentChunk(
                    chunk_id="web-1",
                    source="https://example.com",
                    text="Web research content",
                )
            ]

    class FakeStore:
        def save(self, chunks, embeddings):
            assert chunks == [
                DocumentChunk(
                    chunk_id="web-1",
                    source="https://example.com",
                    text="Web research content",
                )
            ]
            assert embeddings == [[1.0, 2.0]]

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25()
    pipeline.vector_store = None
    pipeline.store = FakeStore()

    assert pipeline.crawl_url(
        "https://example.com",
        max_pages=3,
    ) == 1


def test_pipeline_uses_query_expander():
    calls = []

    class FakeExpander:
        def expand(self, query):
            calls.append(query)
            return [
                "expanded query one",
                "expanded query two",
            ]

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.query_expander = FakeExpander()

    assert pipeline.query_expander.expand("original query") == [
        "expanded query one",
        "expanded query two",
    ]

    assert calls == ["original query"]


def test_pipeline_query_expansion_searches_all_queries(monkeypatch):
    dense_calls = []
    bm25_calls = []

    class FakeExpander:
        def expand(self, query):
            assert query == "original query"
            return [
                "expanded query one",
                "expanded query two",
            ]

    class FakeRetriever:
        def search(
            self,
            query,
            top_k,
            metadata_filter=None,
        ):
            dense_calls.append(query)
            return []

    class FakeBM25:
        def search(
            self,
            query,
            top_k,
            metadata_filter=None,
        ):
            bm25_calls.append(query)
            return []

    class FakeLLM:
        def answer(self, question, evidence):
            return "answer"

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.query_expander = FakeExpander()
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25()
    pipeline.reranker = None
    pipeline.llm = FakeLLM()

    monkeypatch.setattr(
        "research_assistant.pipeline.reciprocal_rank_fusion",
        lambda result_lists, top_k: [],
    )

    assert pipeline.ask(
        "original query",
        use_query_expansion=True,
    ) == "answer"

    assert dense_calls == [
        "original query",
        "expanded query one",
        "expanded query two",
    ]

    assert bm25_calls == [
        "original query",
        "expanded query one",
        "expanded query two",
    ]


def test_pipeline_query_expansion_requires_expander():
    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.query_expander = None

    with pytest.raises(
        ValueError,
        match="query expander is not configured",
    ):
        pipeline.ask(
            "original query",
            use_query_expansion=True,
        )


def test_pipeline_source_verification_requires_verifier():
    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.source_verifier = None

    with pytest.raises(
        ValueError,
        match="source verifier is not configured",
    ):
        pipeline.ask(
            "original query",
            verify_sources=True,
        )


def test_pipeline_filters_unverified_web_sources(monkeypatch):
    class FakeRetriever:
        def search(
            self,
            query,
            top_k,
            metadata_filter=None,
        ):
            return []

    class FakeBM25:
        def search(
            self,
            query,
            top_k,
            metadata_filter=None,
        ):
            return []

    class FakeVerifier:
        def verify(self, source):
            return type(
                "VerificationResult",
                (),
                {"is_valid": source.url == "https://valid.example.com"},
            )()

    class FakeWebProvider:
        def search(self, query, top_k=5):
            return [
                WebSearchResult(
                    title="Valid",
                    url="https://valid.example.com",
                    snippet="valid content",
                ),
                WebSearchResult(
                    title="Invalid",
                    url="https://invalid.example.com",
                    snippet="invalid content",
                ),
            ]

    class FakeLLM:
        def answer(self, question, evidence):
            assert "https://valid.example.com" in evidence
            assert "https://invalid.example.com" not in evidence
            return "answer"

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25()
    pipeline.reranker = None
    pipeline.llm = FakeLLM()
    pipeline.web_search_provider = FakeWebProvider()
    pipeline.source_verifier = FakeVerifier()
    pipeline.query_expander = None

    monkeypatch.setattr(
        "research_assistant.pipeline.reciprocal_rank_fusion",
        lambda result_lists, top_k: [],
    )

    assert pipeline.ask(
        "original query",
        use_web_search=True,
        verify_sources=True,
    ) == "answer"


def test_pipeline_does_not_verify_sources_by_default(monkeypatch):
    class FakeRetriever:
        def search(
            self,
            query,
            top_k,
            metadata_filter=None,
        ):
            return []

    class FakeBM25:
        def search(
            self,
            query,
            top_k,
            metadata_filter=None,
        ):
            return []

    class FakeVerifier:
        def verify(self, source):
            raise AssertionError("source verifier should not be called")

    class FakeWebProvider:
        def search(self, query, top_k=5):
            return [
                WebSearchResult(
                    title="Example",
                    url="https://example.com",
                    snippet="content",
                ),
            ]

    class FakeLLM:
        def answer(self, question, evidence):
            assert "https://example.com" in evidence
            return "answer"

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25()
    pipeline.reranker = None
    pipeline.llm = FakeLLM()
    pipeline.web_search_provider = FakeWebProvider()
    pipeline.source_verifier = FakeVerifier()
    pipeline.query_expander = None

    monkeypatch.setattr(
        "research_assistant.pipeline.reciprocal_rank_fusion",
        lambda result_lists, top_k: [],
    )

    assert pipeline.ask(
        "original query",
        use_web_search=True,
    ) == "answer"


def test_pipeline_research_returns_structured_report():
    class FakeRetriever:
        def search(self, query, top_k=8, metadata_filter=None):
            return [
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id="1",
                        source="docs/example.md",
                        text="Example evidence",
                    ),
                    score=0.9,
                )
            ]

    class FakeLLM:
        def answer(self, question, evidence):
            return "Research answer [1]"

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.llm = FakeLLM()


    class FakeBM25Retriever:
        def search(self, query, top_k=8, metadata_filter=None):
            return []


    pipeline.bm25_retriever = FakeBM25Retriever()
    pipeline.reranker = None
    pipeline.query_expander = None
    pipeline.source_verifier = None
    pipeline.web_search_provider = None

    report = pipeline.research("What is research?")

    assert isinstance(report, ResearchReport)
    assert report.question == "What is research?"
    assert report.answer == "Research answer [1]"
    assert len(report.sources) == 1
    assert report.sources[0].source == "docs/example.md"


def test_pipeline_research_includes_web_sources():
    class FakeRetriever:
        def search(self, query, top_k=8, metadata_filter=None):
            return []

    class FakeBM25Retriever:
        def search(self, query, top_k=8, metadata_filter=None):
            return []

    class FakeLLM:
        def answer(self, question, evidence):
            return "Research answer [1]"

    class FakeWebSearchProvider:
        def search(self, query, top_k=5):
            return [
                WebSearchResult(
                    title="Example Article",
                    url="https://example.com/article",
                    snippet="Example web evidence",
                )
            ]

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25Retriever()
    pipeline.reranker = None
    pipeline.llm = FakeLLM()
    pipeline.query_expander = None
    pipeline.source_verifier = None
    pipeline.web_search_provider = FakeWebSearchProvider()

    report = pipeline.research(
        "What is research?",
        use_web_search=True,
    )

    assert len(report.sources) == 1
    assert report.sources[0].source == "https://example.com/article"
    assert report.sources[0].title == "Example Article"
    assert report.sources[0].url == "https://example.com/article"


def test_pipeline_research_filters_unverified_web_sources():
    class FakeRetriever:
        def search(self, query, top_k=8, metadata_filter=None):
            return []

    class FakeBM25Retriever:
        def search(self, query, top_k=8, metadata_filter=None):
            return []

    class FakeLLM:
        def answer(self, question, evidence):
            return "Research answer [1]"

    class FakeWebSearchProvider:
        def search(self, query, top_k=5):
            return [
                WebSearchResult(
                    title="Valid Article",
                    url="https://example.com/valid",
                    snippet="Valid evidence",
                ),
                WebSearchResult(
                    title="Invalid Article",
                    url="https://example.com/invalid",
                    snippet="Invalid evidence",
                ),
            ]

    class FakeSourceVerifier:
        def verify(self, source):
            return SourceVerificationResult(
                source=source,
                is_valid=source.url.endswith("/valid"),
            )

    pipeline = ResearchPipeline.__new__(ResearchPipeline)
    pipeline.retriever = FakeRetriever()
    pipeline.bm25_retriever = FakeBM25Retriever()
    pipeline.reranker = None
    pipeline.llm = FakeLLM()
    pipeline.query_expander = None
    pipeline.web_search_provider = FakeWebSearchProvider()
    pipeline.source_verifier = FakeSourceVerifier()

    report = pipeline.research(
        "What is research?",
        use_web_search=True,
        verify_sources=True,
    )

    assert len(report.sources) == 1
    assert report.sources[0].title == "Valid Article"
    assert report.sources[0].url == "https://example.com/valid"