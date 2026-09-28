from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    index_dir: str | Path = "data/index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_reranker: bool = True