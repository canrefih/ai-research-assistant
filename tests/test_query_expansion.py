import pytest

from research_assistant.query_expansion import QueryExpander


def test_query_expander_is_abstract():
    with pytest.raises(TypeError):
        QueryExpander()


def test_llm_query_expander_returns_three_queries():
    class FakeLLM:
        def answer(self, question, evidence):
            return (
                "semantic retrieval methods\n"
                "dense vector search techniques\n"
                "hybrid retrieval approaches\n"
                "extra query"
            )

    from research_assistant.query_expansion import LLMQueryExpander

    expander = LLMQueryExpander(llm=FakeLLM())

    assert expander.expand("How does retrieval work?") == [
        "semantic retrieval methods",
        "dense vector search techniques",
        "hybrid retrieval approaches",
    ]


def test_llm_query_expander_rejects_empty_query():
    from research_assistant.query_expansion import LLMQueryExpander

    expander = LLMQueryExpander(llm=object())

    with pytest.raises(ValueError, match="query must not be empty"):
        expander.expand("   ")