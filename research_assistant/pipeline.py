from pathlib import Path

from .ingestion import load_directory
from .llm import LLMClient
from .retrieval import Reranker, SemanticRetriever
from .storage import IndexStore


class ResearchPipeline:
    def __init__(
        self,
        use_reranker: bool = True,
        index_dir: str | Path = "data/index",
    ):
        self.retriever = SemanticRetriever()
        self.reranker = Reranker() if use_reranker else None
        self.llm = LLMClient()
        self.store = IndexStore(index_dir)

    def index(self, directory: str | Path) -> int:
        chunks = load_directory(directory)
        if not chunks:
            raise ValueError(f"No .md or .txt documents found in {directory}")
        self.retriever.fit(chunks)
        self.store.save(chunks, self.retriever.embeddings)
        return len(chunks)

    def load_index(self) -> int:
        chunks, embeddings = self.store.load()
        self.retriever.load(chunks, embeddings)
        return len(chunks)

    def ask(self, question: str, top_k: int = 8) -> str:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        results = self.retriever.search(question, top_k=top_k)
        if self.reranker:
            results = self.reranker.rerank(
                question, results, top_k=min(5, len(results))
            )

        evidence = "\n\n".join(
            f"[{i}] Source: {r.chunk.source}\n{r.chunk.text}"
            for i, r in enumerate(results, 1)
        )
        return self.llm.answer(question, evidence)
