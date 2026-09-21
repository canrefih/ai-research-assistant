import os
from abc import ABC, abstractmethod

import requests

from .models import WebSearchResult


class WebSearchProvider(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[WebSearchResult]:
        raise NotImplementedError


class TavilySearchProvider(WebSearchProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[WebSearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": top_k,
            },
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()

        return [
            WebSearchResult(
                title=result["title"],
                url=result["url"],
                snippet=result.get("content", ""),
            )
            for result in data.get("results", [])
        ]