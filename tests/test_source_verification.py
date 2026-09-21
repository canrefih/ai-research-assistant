import pytest
import requests

from research_assistant.models import WebSearchResult
from research_assistant.source_verification import (
    HttpSourceVerifier,
    SourceVerifier,
)


def test_source_verifier_is_abstract():
    with pytest.raises(TypeError):
        SourceVerifier()


def test_http_source_verifier_accepts_successful_response(monkeypatch):
    source = WebSearchResult(
        title="Example",
        url="https://example.com",
        snippet="Example content",
    )

    class FakeResponse:
        status_code = 200

    def fake_head(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "research_assistant.source_verification.requests.head",
        fake_head,
    )

    result = HttpSourceVerifier().verify(source)

    assert result.source == source
    assert result.is_valid is True
    assert result.status_code == 200
    assert result.error is None


def test_http_source_verifier_rejects_error_response(monkeypatch):
    source = WebSearchResult(
        title="Example",
        url="https://example.com/missing",
        snippet="Example content",
    )

    class FakeResponse:
        status_code = 404

    def fake_head(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "research_assistant.source_verification.requests.head",
        fake_head,
    )

    result = HttpSourceVerifier().verify(source)

    assert result.source == source
    assert result.is_valid is False
    assert result.status_code == 404
    assert result.error is None


def test_http_source_verifier_handles_request_error(monkeypatch):
    source = WebSearchResult(
        title="Example",
        url="https://example.com",
        snippet="Example content",
    )

    def fake_head(*args, **kwargs):
        raise requests.RequestException("connection failed")

    monkeypatch.setattr(
        "research_assistant.source_verification.requests.head",
        fake_head,
    )

    result = HttpSourceVerifier().verify(source)

    assert result.source == source
    assert result.is_valid is False
    assert result.status_code is None
    assert result.error == "connection failed"


def test_http_source_verifier_rejects_invalid_timeout():
    with pytest.raises(
        ValueError,
        match="timeout must be greater than 0",
    ):
        HttpSourceVerifier(timeout=0)