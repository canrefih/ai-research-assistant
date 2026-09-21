from research_assistant.models import WebSearchResult


def test_web_search_result():
    result = WebSearchResult(
        title="RAG Research",
        url="https://example.com/rag",
        snippet="Retrieval augmented generation research.",
    )

    assert result.title == "RAG Research"
    assert result.url == "https://example.com/rag"
    assert result.snippet == "Retrieval augmented generation research."