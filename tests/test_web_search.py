import pytest
import requests

from research_assistant.models import WebSearchResult
from research_assistant.web_search import (
    TavilySearchProvider,
    WebSearchProvider,
)


def test_web_search_provider_requires_search_implementation():
    with pytest.raises(TypeError):
        WebSearchProvider()


def test_tavily_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TAVILY_API_KEY is required"):
        TavilySearchProvider()


def test_tavily_provider_accepts_api_key():
    provider = TavilySearchProvider(
        api_key="test-api-key",
    )

    assert provider.api_key == "test-api-key"


def test_tavily_provider_reads_api_key_from_environment(monkeypatch):
    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "env-test-api-key",
    )

    provider = TavilySearchProvider()

    assert provider.api_key == "env-test-api-key"


def test_tavily_search_maps_results(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {
                        "title": "RAG Research",
                        "url": "https://example.com/rag",
                        "content": "Retrieval augmented generation research.",
                    },
                    {
                        "title": "Vector Search",
                        "url": "https://example.com/vector",
                        "content": "Semantic vector search overview.",
                    },
                ]
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "research_assistant.web_search.requests.post",
        fake_post,
    )

    provider = TavilySearchProvider(api_key="test-api-key")

    results = provider.search(
        "RAG research",
        top_k=2,
    )

    assert results == [
        WebSearchResult(
            title="RAG Research",
            url="https://example.com/rag",
            snippet="Retrieval augmented generation research.",
        ),
        WebSearchResult(
            title="Vector Search",
            url="https://example.com/vector",
            snippet="Semantic vector search overview.",
        ),
    ]

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["kwargs"] == {
        "json": {
            "api_key": "test-api-key",
            "query": "RAG research",
            "max_results": 2,
        },
        "timeout": 10,
    }


def test_tavily_search_rejects_empty_query():
    provider = TavilySearchProvider(api_key="test-api-key")

    with pytest.raises(ValueError, match="query must not be empty"):
        provider.search("   ")


def test_tavily_search_rejects_invalid_top_k():
    provider = TavilySearchProvider(api_key="test-api-key")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        provider.search("RAG", top_k=0)


def test_tavily_search_propagates_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("Tavily request failed")

    def fake_post(url, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "research_assistant.web_search.requests.post",
        fake_post,
    )

    provider = TavilySearchProvider(api_key="test-api-key")

    with pytest.raises(requests.HTTPError, match="Tavily request failed"):
        provider.search("RAG")