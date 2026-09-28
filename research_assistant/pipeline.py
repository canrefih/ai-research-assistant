import logging
from pathlib import Path

from research_assistant.query_expansion import QueryExpander
from .vector_store import VectorStoreProtocol

from .ingestion import load_directory, load_url
from .llm import LLMClient
from .models import ResearchReport, ResearchSource
from .retrieval import (
    BM25Retriever,
    Reranker,
    SemanticRetriever,
    reciprocal_rank_fusion,
)
from .storage import IndexStore, IndexStoreProtocol
from .web_search import WebSearchProvider
from .source_verification import SourceVerifier


logger = logging.getLogger(__name__)

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
        source_verifier: SourceVerifier | None = None,
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
        self.source_verifier = source_verifier
    def index(self, directory: str | Path) -> int:
        logger.info("Indexing directory: %s", directory)
        chunks = load_directory(directory)
        if not chunks:
            raise ValueError(f"No .md or .txt documents found in {directory}")
        self.retriever.fit(chunks)
        self.bm25_retriever.fit(chunks)

        if self.vector_store is None:
            self.store.save(chunks, self.retriever.embeddings)

        logger.info("Indexed %d chunks from %s", len(chunks), directory)
        return len(chunks)

    def index_url(self, url: str) -> int:
        logger.info("Indexing URL: %s", url)
        chunks = load_url(url)
        if not chunks:
            raise ValueError(f"No content found at {url}")
        self.retriever.fit(chunks)
        self.bm25_retriever.fit(chunks)

        if self.vector_store is None:
            self.store.save(chunks, self.retriever.embeddings)

        logger.info("Indexed %d chunks from URL: %s", len(chunks), url)
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
        logger.info("Loading index")
        if self.vector_store is not None:
            chunks = self.vector_store.load_chunks()
            self.bm25_retriever.fit(chunks)
            logger.info("Loaded %d chunks from vector store", len(chunks))
            return len(chunks)

        chunks, embeddings = self.store.load()
        self.retriever.load(chunks, embeddings)
        self.bm25_retriever.fit(chunks)
        logger.info("Loaded %d chunks from local index", len(chunks))
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
        verify_sources: bool = False,
    ) -> str:
        evidence, _ = self._build_research_context(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            use_web_search=use_web_search,
            use_query_expansion=use_query_expansion,
            verify_sources=verify_sources,
        )

        return self.llm.answer(question, evidence)

    def research(
        self,
        question: str,
        top_k: int = 8,
        metadata_filter: dict[str, str] | None = None,
        use_web_search: bool = False,
        use_query_expansion: bool = False,
        verify_sources: bool = False,
    ) -> ResearchReport:
        evidence, sources = self._build_research_context(
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
            use_web_search=use_web_search,
            use_query_expansion=use_query_expansion,
            verify_sources=verify_sources,
        )

        answer = self.llm.answer(question, evidence)

        return ResearchReport(
            question=question,
            answer=answer,
            sources=sources,
        )

    def _build_research_context(
        self,
        question: str,
        top_k: int = 8,
        metadata_filter: dict[str, str] | None = None,
        use_web_search: bool = False,
        use_query_expansion: bool = False,
        verify_sources: bool = False,
    ) -> tuple[str, list[ResearchSource]]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if use_query_expansion and self.query_expander is None:
            raise ValueError("query expander is not configured")
        if verify_sources and self.source_verifier is None:
            raise ValueError("source verifier is not configured")

        logger.info(
            "Building research context: top_k=%d, web_search=%s, "
            "query_expansion=%s, verify_sources=%s",
            top_k,
            use_web_search,
            use_query_expansion,
            verify_sources,
        )
        queries = [question]

        if use_query_expansion:
            queries.extend(
                self.query_expander.expand(question)
            )

        logger.info("Retrieving with %d query variant(s)", len(queries))
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

        logger.info("Retrieved %d fused results", len(results))
        if self.reranker:
            results = self.reranker.rerank(
                question,
                results,
                top_k=min(top_k, len(results)),
            )
            logger.info("Reranked results: %d", len(results))

        evidence_parts = [
            f"[{i}] Source: {r.chunk.source}\n{r.chunk.text}"
            for i, r in enumerate(results, 1)
        ]

        sources = [
            ResearchSource(source=r.chunk.source)
            for r in results
        ]

        if use_web_search:
            web_results = self.search_web(
                question,
                top_k=top_k,
            )

            logger.info("Web search returned %d results", len(web_results))
            if verify_sources:
                web_results = [
                    result
                    for result in web_results
                    if self.source_verifier.verify(result).is_valid
                ]
                logger.info(
                    "Source verification kept %d web results",
                    len(web_results),
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

            sources.extend(
                ResearchSource(
                    source=result.url,
                    title=result.title,
                    url=result.url,
                )
                for result in web_results
            )

        return "\n\n".join(evidence_parts), sources