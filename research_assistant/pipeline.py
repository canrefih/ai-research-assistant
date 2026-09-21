from pathlib import Path
from .vector_store import VectorStoreProtocol

from .ingestion import load_directory, load_url
from .llm import LLMClient
from .retrieval import (
    BM25Retriever,
    Reranker,
    SemanticRetriever,
    reciprocal_rank_fusion,
)
from .storage import IndexStore, IndexStoreProtocol


class ResearchPipeline:
    def __init__(
        self,
        use_reranker: bool = True,
        index_dir: str | Path = "data/index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        store: IndexStoreProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
    ):
        self.retriever = SemanticRetriever(
            model_name=embedding_model,
            vector_store=vector_store,
        )
        self.vector_store = vector_store
        self.bm25_retriever = BM25Retriever()
        self.reranker = (
            Reranker(model_name=reranker_model)
            if use_reranker
            else None
        )
        self.llm = LLMClient()
        self.store = store or IndexStore(index_dir)

    def index(self, directory: str | Path) -> int:
        chunks = load_directory(directory)
        if not chunks:
            raise ValueError(f"No .md or .txt documents found in {directory}")
        self.retriever.fit(chunks)
        self.bm25_retriever.fit(chunks)

        if self.vector_store is None:
            self.store.save(chunks, self.retriever.embeddings)

        return len(chunks)

    def index_url(self, url: str) -> int:
        chunks = load_url(url)
        if not chunks:
            raise ValueError(f"No content found at {url}")
        self.retriever.fit(chunks)
        self.bm25_retriever.fit(chunks)

        if self.vector_store is None:
            self.store.save(chunks, self.retriever.embeddings)

        return len(chunks)

    def load_index(self) -> int:
        if self.vector_store is not None:
            chunks = self.vector_store.load_chunks()
            self.bm25_retriever.fit(chunks)
            return len(chunks)

        chunks, embeddings = self.store.load()
        self.retriever.load(chunks, embeddings)
        self.bm25_retriever.fit(chunks)
        return len(chunks)

    def close(self) -> None:
        if self.vector_store is not None:
            self.vector_store.close()

    def ask(
        self,
        question: str,
        top_k: int = 8,
        metadata_filter: dict[str, str] | None = None,
    ) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        dense_results = self.retriever.search(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        bm25_results = self.bm25_retriever.search(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        results = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            top_k=top_k,
        )

        if self.reranker:
            results = self.reranker.rerank(
                question,
                results,
                top_k=min(top_k, len(results)),
            )

        evidence = "\n\n".join(
            f"[{i}] Source: {r.chunk.source}\n{r.chunk.text}"
            for i, r in enumerate(results, 1)
        )
        return self.llm.answer(question, evidence)
