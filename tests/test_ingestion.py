from pathlib import Path

from research_assistant.ingestion import load_directory

def test_load_directory(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nSemantic retrieval is useful.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Python is widely used for ML.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("a,b", encoding="utf-8")
    chunks = load_directory(tmp_path)
    assert len(chunks) == 2


def test_load_directory_adds_file_metadata(tmp_path):
    document = tmp_path / "python.md"
    document.write_text(
        "# Python\n\nPython is a programming language.",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "python.md",
        "file_type": "md",
    }


def test_load_directory_adds_txt_file_metadata(tmp_path):
    document = tmp_path / "notes.txt"
    document.write_text(
        "Python is useful for research.",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "notes.txt",
        "file_type": "txt",
    }


def test_load_directory_deduplicates_identical_files(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    content = "# Python\n\nPython is a programming language."

    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1


def test_load_directory_keeps_different_files(tmp_path):
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"

    first.write_text(
        "# Python\n\nPython is a programming language.",
        encoding="utf-8",
    )
    second.write_text(
        "# Java\n\nJava is a programming language.",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 2


def test_load_directory_deduplicates_same_content_with_different_metadata(
    tmp_path,
):
    first = tmp_path / "first.md"
    second = tmp_path / "second.txt"

    content = "Python is a programming language."

    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].metadata == {
        "source": "first.md",
        "file_type": "md",
    }


def test_load_directory_reads_pdf(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return "PDF research test"

    class FakeReader:
        def __init__(self, path):
            self.path = path
            self.pages = [FakePage()]

    monkeypatch.setattr(
        "research_assistant.ingestion.PdfReader",
        FakeReader,
    )

    document = tmp_path / "research.pdf"
    document.write_bytes(b"fake pdf")

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].text == "PDF research test"
    assert chunks[0].metadata == {
        "source": "research.pdf",
        "file_type": "pdf",
    }


def test_load_directory_reads_html_and_removes_non_content_elements(tmp_path):
    document = tmp_path / "research.html"
    document.write_text(
        """
        <html>
            <head>
                <title>Research</title>
                <style>.hidden { display: none; }</style>
                <script>alert("ignore me");</script>
            </head>
            <body>
                <h1>Research Assistant</h1>
                <p>Semantic retrieval is useful.</p>
                <noscript>Fallback content</noscript>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert "Research Assistant" in chunks[0].text
    assert "Semantic retrieval is useful." in chunks[0].text
    assert "ignore me" not in chunks[0].text
    assert "display: none" not in chunks[0].text
    assert "Fallback content" not in chunks[0].text
    assert chunks[0].metadata == {
        "source": "research.html",
        "file_type": "html",
    }


def test_load_directory_reads_htm(tmp_path):
    document = tmp_path / "research.htm"
    document.write_text(
        "<html><body><h1>HTM research</h1></body></html>",
        encoding="utf-8",
    )

    chunks = load_directory(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].text == "HTM research"
    assert chunks[0].metadata == {
        "source": "research.htm",
        "file_type": "htm",
    }


def test_fetch_html(monkeypatch):
    class FakeResponse:
        text = "<html><body><h1>Hello</h1></body></html>"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        assert url == "https://example.com"
        assert timeout == 10
        return FakeResponse()

    monkeypatch.setattr(
        "research_assistant.ingestion.requests.get",
        fake_get,
    )

    from research_assistant.ingestion import _fetch_html

    assert _fetch_html("https://example.com") == (
        "<html><body><h1>Hello</h1></body></html>"
    )


def test_parse_html_removes_non_content_elements():
    from research_assistant.ingestion import _parse_html

    html = """
    <html>
        <head>
            <style>.hidden { display: none; }</style>
            <script>alert("ignore me");</script>
        </head>
        <body>
            <h1>Research</h1>
            <p>Useful content.</p>
            <noscript>Fallback</noscript>
        </body>
    </html>
    """

    text = _parse_html(html)

    assert "Research" in text
    assert "Useful content." in text
    assert "ignore me" not in text
    assert "display: none" not in text
    assert "Fallback" not in text


def test_fetch_web_page(monkeypatch):
    from research_assistant.ingestion import fetch_web_page

    def fake_fetch_html(url):
        assert url == "https://example.com"
        return """
        <html>
            <body>
                <h1>Research</h1>
                <p>Useful content.</p>
                <script>alert("ignore me");</script>
            </body>
        </html>
        """

    monkeypatch.setattr(
        "research_assistant.ingestion._fetch_html",
        fake_fetch_html,
    )

    text = fetch_web_page("https://example.com")

    assert "Research" in text
    assert "Useful content." in text
    assert "ignore me" not in text


def test_load_url_creates_chunks(monkeypatch):
    from research_assistant.ingestion import load_url

    def fake_fetch_web_page(url):
        assert url == "https://example.com/research"
        return "Semantic retrieval is useful for research."

    monkeypatch.setattr(
        "research_assistant.ingestion.fetch_web_page",
        fake_fetch_web_page,
    )

    chunks = load_url("https://example.com/research")

    assert len(chunks) == 1
    assert chunks[0].text == "Semantic retrieval is useful for research."
    assert chunks[0].metadata == {
        "source": "https://example.com/research",
        "file_type": "html",
    }


def test_fetch_html_raises_for_http_error(monkeypatch):
    import pytest
    import requests

    from research_assistant.ingestion import _fetch_html

    class FakeResponse:
        text = ""

        def raise_for_status(self):
            raise requests.HTTPError("404 Client Error")

    def fake_get(url, timeout):
        return FakeResponse()

    monkeypatch.setattr(
        "research_assistant.ingestion.requests.get",
        fake_get,
    )

    with pytest.raises(requests.HTTPError, match="404 Client Error"):
        _fetch_html("https://example.com/missing")


def test_fetch_html_propagates_timeout(monkeypatch):
    import requests

    from research_assistant.ingestion import _fetch_html

    def fake_get(url, timeout):
        assert timeout == 10
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(
        "research_assistant.ingestion.requests.get",
        fake_get,
    )

    try:
        _fetch_html("https://example.com")
    except requests.Timeout as exc:
        assert str(exc) == "Request timed out"
    else:
        raise AssertionError("Expected requests.Timeout")