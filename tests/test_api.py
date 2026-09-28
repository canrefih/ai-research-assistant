from fastapi.testclient import TestClient

from research_assistant.api import app, get_pipeline
from research_assistant.pipeline import ResearchPipeline


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_pipeline_returns_research_pipeline():
    pipeline = get_pipeline()

    assert isinstance(pipeline, ResearchPipeline)


def test_get_pipeline_returns_cached_instance():
    first = get_pipeline()
    second = get_pipeline()

    assert first is second


def test_ask():
    class FakeReport:
        answer = "Python is a programming language."
        sources = [
            type(
                "Source",
                (),
                {
                    "source": "python.md",
                    "title": "Python",
                    "url": "https://example.com/python",
                },
            )(),
        ]

    class FakePipeline:
        def ask(self, question, use_web_search=False):
            assert question == "What is Python?"
            assert use_web_search is False
            return FakeReport()

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    try:
        response = client.post(
            "/ask",
            json={"question": "What is Python?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Python is a programming language.",
        "sources": [
            {
                "source": "python.md",
                "title": "Python",
                "url": "https://example.com/python",
            },
        ],
    }


def test_ask_passes_web_search_option():
    class FakeReport:
        answer = "Web-enabled answer."
        sources = []

    class FakePipeline:
        def ask(self, question, use_web_search=False):
            assert question == "What is Python?"
            assert use_web_search is True
            return FakeReport()

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    try:
        response = client.post(
            "/ask",
            json={
                "question": "What is Python?",
                "use_web_search": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Web-enabled answer.",
        "sources": [],
    }


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_returns_bad_request_for_pipeline_value_error():
    class FakePipeline:
        def ask(self, question, use_web_search=False):
            raise ValueError("Index is not available")

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    try:
        response = client.post(
            "/ask",
            json={"question": "What is Python?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Index is not available",
    }


def test_ask_returns_bad_request_for_unfitted_retriever():
    class FakePipeline:
        def ask(self, question, use_web_search=False):
            raise RuntimeError("Retriever is not fitted")

    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()

    try:
        response = client.post(
            "/ask",
            json={"question": "What is Python?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Retriever is not fitted",
    }