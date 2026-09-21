from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests

from .models import WebSearchResult


@dataclass(frozen=True)
class SourceVerificationResult:
    source: WebSearchResult
    is_valid: bool
    status_code: int | None = None
    error: str | None = None


class SourceVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        source: WebSearchResult,
    ) -> SourceVerificationResult:
        raise NotImplementedError


class HttpSourceVerifier(SourceVerifier):
    def __init__(self, timeout: float = 10):
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        self.timeout = timeout

    def verify(
        self,
        source: WebSearchResult,
    ) -> SourceVerificationResult:
        try:
            response = requests.head(
                source.url,
                timeout=self.timeout,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            return SourceVerificationResult(
                source=source,
                is_valid=False,
                error=str(exc),
            )

        return SourceVerificationResult(
            source=source,
            is_valid=200 <= response.status_code < 400,
            status_code=response.status_code,
        )