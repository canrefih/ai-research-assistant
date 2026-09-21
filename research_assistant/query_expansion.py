from abc import ABC, abstractmethod
from .llm import LLMClient


class QueryExpander(ABC):
    @abstractmethod
    def expand(self, query: str) -> list[str]:
        raise NotImplementedError


class LLMQueryExpander(QueryExpander):
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def expand(self, query: str) -> list[str]:
        if not query.strip():
            raise ValueError("query must not be empty")

        prompt = (
            "Generate 3 alternative search queries for the following query. "
            "Return exactly one query per line and nothing else.\n\n"
            f"Query: {query}"
        )

        response = self.llm.answer(
            prompt,
            "",
        )

        queries = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        return queries[:3]