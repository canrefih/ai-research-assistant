from pathlib import Path

from research_assistant.query_expansion import QueryExpander
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
from .web_search import WebSearchProvider


class ResearchPipeline:
    def __init__(
        self,
        use_reranker: bool = True,
        index_dir: str | Path = "data/index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        store: IndexStoreProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        web_search_provider: WebSearchProvider | None = None,
        query_expander: QueryExpander | None = None,
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
        self.web_search_provider = web_search_provider
        self.query_expander = query_expander

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

    def crawl_url(
        self,
        url: str,
        max_pages: int = 5,
    ) -> int:
        from .crawler import WebCrawler

        crawler = WebCrawler(max_pages=max_pages)
        chunks = crawler.crawl_chunks(url)

        if not chunks:
            raise ValueError(f"No content found while crawling {url}")

        self.retriever.fit(chunks)
        self.bm25_retriever.fit(chunks)

        if self.vector_store is None:
            self.store.save(chunks, self.retriever.embeddings)

        return len(chunks)

    def search_web(
        self,
        query: str,
        top_k: int = 5,
    ):
        if self.web_search_provider is None:
            raise ValueError("web search provider is not configured")

        return self.web_search_provider.search(
            query,
            top_k=top_k,
        )

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
        use_web_search: bool = False,
        use_query_expansion: bool = False,
    ) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if use_query_expansion and self.query_expander is None:
            raise ValueError("query expander is not configured")

        queries = [question]

        if use_query_expansion:
            queries.extend(
                self.query_expander.expand(question)
            )

        dense_results = [
            self.retriever.search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            for query in queries
        ]

        bm25_results = [
            self.bm25_retriever.search(
                query,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
            for query in queries
        ]

        results = reciprocal_rank_fusion(
            dense_results + bm25_results,
            top_k=top_k,
        )

        if self.reranker:
            results = self.reranker.rerank(
                question,
                results,
                top_k=min(top_k, len(results)),
            )

        evidence_parts = [
            f"[{i}] Source: {r.chunk.source}\n{r.chunk.text}"
            for i, r in enumerate(results, 1)
        ]

        if use_web_search:
            web_results = self.search_web(
                question,
                top_k=top_k,
            )

            start_index = len(evidence_parts) + 1

            evidence_parts.extend(
                f"[{i}] Source: {result.url}\n"
                f"Title: {result.title}\n"
                f"{result.snippet}"
                for i, result in enumerate(
                    web_results,
                    start_index,
                )
            )

        evidence = "\n\n".join(evidence_parts)
        return self.llm.answer(question, evidence)
