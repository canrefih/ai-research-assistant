import pytest

from research_assistant.crawler import WebCrawler


def test_crawler_requires_positive_max_pages():
    with pytest.raises(ValueError, match="max_pages"):
        WebCrawler(max_pages=0)


def test_crawler_rejects_invalid_start_url():
    crawler = WebCrawler()

    with pytest.raises(
        ValueError,
        match="valid HTTP or HTTPS URL",
    ):
        crawler.crawl("not-a-url")


def test_crawler_follows_same_domain_links(monkeypatch):
    pages = {
        "https://example.com/": """
            <html>
                <body>
                    <h1>Home</h1>
                    <a href="/about">About</a>
                    <a href="https://other.example.com/page">External</a>
                </body>
            </html>
        """,
        "https://example.com/about": """
            <html>
                <body>
                    <h1>About</h1>
                    <a href="/">Home</a>
                </body>
            </html>
        """,
    }

    def fake_fetch(url):
        return pages[url]

    monkeypatch.setattr(
        "research_assistant.crawler._fetch_html",
        fake_fetch,
    )

    crawler = WebCrawler(max_pages=5)

    result = crawler.crawl("https://example.com/")

    assert result == [
        ("https://example.com/", "Home About External"),
        ("https://example.com/about", "About Home"),
    ]


def test_crawler_respects_max_pages(monkeypatch):
    pages = {
        "https://example.com/": """
            <a href="/one">One</a>
            <a href="/two">Two</a>
        """,
        "https://example.com/one": "One page",
        "https://example.com/two": "Two page",
    }

    def fake_fetch(url):
        return pages[url]

    monkeypatch.setattr(
        "research_assistant.crawler._fetch_html",
        fake_fetch,
    )

    crawler = WebCrawler(max_pages=2)

    result = crawler.crawl("https://example.com/")

    assert len(result) == 2
    assert result[0][0] == "https://example.com/"
    assert result[1][0] == "https://example.com/one"


def test_crawler_does_not_visit_duplicate_urls(monkeypatch):
    pages = {
        "https://example.com/": """
            <a href="/page">Page</a>
            <a href="https://example.com/page#section">Same page</a>
        """,
        "https://example.com/page": "Page content",
    }

    fetch_count = {}

    def fake_fetch(url):
        fetch_count[url] = fetch_count.get(url, 0) + 1
        return pages[url]

    monkeypatch.setattr(
        "research_assistant.crawler._fetch_html",
        fake_fetch,
    )

    crawler = WebCrawler(max_pages=5)

    result = crawler.crawl("https://example.com/")

    assert len(result) == 2
    assert fetch_count == {
        "https://example.com/": 1,
        "https://example.com/page": 1,
    }


def test_crawler_skips_non_http_links(monkeypatch):
    html = """
        <a href="mailto:test@example.com">Email</a>
        <a href="javascript:void(0)">JavaScript</a>
        <a href="/valid">Valid</a>
    """

    pages = {
        "https://example.com/": html,
        "https://example.com/valid": "Valid page",
    }

    monkeypatch.setattr(
        "research_assistant.crawler._fetch_html",
        lambda url: pages[url],
    )

    crawler = WebCrawler(max_pages=5)

    result = crawler.crawl("https://example.com/")

    assert [url for url, _ in result] == [
        "https://example.com/",
        "https://example.com/valid",
    ]


def test_crawler_chunks_pages(monkeypatch):
    pages = {
        "https://example.com/": """
            <h1>Home</h1>
            <p>This is the home page.</p>
            <a href="/about">About</a>
        """,
        "https://example.com/about": """
            <h1>About</h1>
            <p>This is the about page.</p>
        """,
    }

    monkeypatch.setattr(
        "research_assistant.crawler._fetch_html",
        lambda url: pages[url],
    )

    crawler = WebCrawler(max_pages=5)

    chunks = crawler.crawl_chunks("https://example.com/")

    assert len(chunks) == 2
    assert chunks[0].source == "https://example.com/"
    assert chunks[0].metadata == {
        "source": "https://example.com/",
        "file_type": "html",
    }
    assert chunks[1].source == "https://example.com/about"