from .ingestion import load_directory
from .llm import LLMClient
from .retrieval import SemanticRetriever, Reranker

class ResearchPipeline:
    def __init__(self, use_reranker: bool = True):
        self.retriever = SemanticRetriever()
        self.reranker = Reranker() if use_reranker else None
        self.llm = LLMClient()

    def index(self, directory: str) -> int:
        chunks = load_directory(directory)
        self.retriever.fit(chunks)
        return len(chunks)

    def ask(self, question: str, top_k: int = 8) -> str:
        results = self.retriever.search(question, top_k=top_k)
        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=min(5, len(results)))

        evidence = "\n\n".join(
            f"[{i}] Source: {r.chunk.source}\n{r.chunk.text}"
            for i, r in enumerate(results, 1)
        )
        return self.llm.answer(question, evidence)
