from collections import deque
from urllib.parse import urljoin, urlparse

from .ingestion import _fetch_html, _parse_html
from .models import DocumentChunk


class WebCrawler:
    def __init__(self, max_pages: int = 5):
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")

        self.max_pages = max_pages

    def crawl(self, start_url: str) -> list[tuple[str, str]]:
        self._validate_url(start_url)

        queue = deque([start_url])
        queued = {start_url}
        visited: set[str] = set()
        pages: list[tuple[str, str]] = []

        domain = urlparse(start_url).netloc

        while queue and len(pages) < self.max_pages:
            url = queue.popleft()

            if url in visited:
                continue

            visited.add(url)

            html = _fetch_html(url)
            text = _parse_html(html)

            pages.append((url, text))

            if len(pages) >= self.max_pages:
                break

            for link in self._extract_links(html, url):
                if link in visited or link in queued:
                    continue

                if urlparse(link).netloc != domain:
                    continue

                queue.append(link)
                queued.add(link)

        return pages

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("start_url must be a valid HTTP or HTTPS URL")

    @staticmethod
    def _extract_links(html: str, base_url: str) -> list[str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        links: list[str] = []

        for anchor in soup.find_all("a", href=True):
            absolute_url = urljoin(
                base_url,
                anchor["href"],
            )

            parsed = urlparse(absolute_url)

            if parsed.scheme not in {"http", "https"}:
                continue

            normalized = parsed._replace(
                fragment="",
            ).geturl()

            if normalized not in links:
                links.append(normalized)

        return links


    def crawl_chunks(self, start_url: str) -> list[DocumentChunk]:
        from .chunking import chunk_text

        pages = self.crawl(start_url)
        chunks = []

        for url, text in pages:
            chunks.extend(
                chunk_text(
                    text,
                    url,
                    metadata={
                        "source": url,
                        "file_type": "html",
                    },
                )
            )

        return chunks