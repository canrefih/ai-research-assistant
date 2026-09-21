# AI Research Assistant

A local-first, citation-aware research assistant for turning documents and web pages into searchable evidence and grounded LLM answers.

## What it does

The project implements a practical hybrid retrieval and reranking pipeline:

1. Ingest Markdown, text, PDF, HTML documents, and web pages while preserving their sources.
2. Split documents into overlapping chunks.
3. Create dense embeddings with Sentence Transformers.
4. Persist chunks and embeddings to either a local NumPy index or Qdrant.
5. Retrieve candidates using both dense semantic search and BM25.
6. Combine retrieval results with reciprocal rank fusion.
7. Optionally rerank candidates with a CrossEncoder.
8. Send only the retrieved evidence to an OpenAI-compatible chat endpoint.
9. Ask the model to cite the supplied sources as [1], [2], etc.

This keeps the core small while leaving clear extension points for web search, query expansion, evaluation, source verification, and a UI.

## Architecture

    Documents
        |
        v
    Ingestion -> Chunking + source metadata
        |
        v
    Sentence Transformer embeddings
        |
        +----------------------+
        |                      |
        v                      v
    Vector store           BM25 index
    (NumPy / Qdrant)           |
        |                      |
        +----------+-----------+
                   |
                   v
        Reciprocal rank fusion
                   |
                   v
        Optional CrossEncoder reranking
                   |
                   v
            Evidence context
                   |
                   v
        OpenAI-compatible LLM
                   |
                   v
        Answer + source citations

## Features

- Markdown and plain-text ingestion
- PDF ingestion
- HTML/HTM ingestion
- Web page URL ingestion
- Tavily web search provider
- Overlapping chunking with source metadata
- Dense semantic retrieval
- BM25 lexical retrieval
- Hybrid dense + BM25 retrieval with reciprocal rank fusion
- Optional CrossEncoder reranking
- Persistent local NumPy index
- Qdrant vector store with local persistence
- Metadata filtering across dense and lexical retrieval
- OpenAI-compatible LLM endpoint
- Citation-oriented evidence formatting
- CLI interface
- Unit tests
- Environment-based API configuration
- No secrets committed to the repository

## Requirements

- Python 3.10+
- Internet access on first run to download the embedding/reranker models
- An OpenAI-compatible LLM endpoint is optional

Default embedding model:

    sentence-transformers/all-MiniLM-L6-v2

Optional reranker:

    cross-encoder/ms-marco-MiniLM-L6-v2

## Installation

    git clone https://github.com/canrefih/ai-research-assistant.git
    cd ai-research-assistant

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

On Windows PowerShell:

    .venv\Scripts\Activate.ps1
    pip install -e ".[dev]"

## Configuration

Copy the example environment file:

    cp .env.example .env

Then configure an OpenAI-compatible endpoint:

    LLM_BASE_URL=https://api.openai.com/v1
    LLM_API_KEY=your-api-key
    LLM_MODEL=your-model-name
    TAVILY_API_KEY=your-tavily-api-key

The client expects the standard chat-completions route:

    POST {LLM_BASE_URL}/chat/completions

The LLM is optional. Without configuration, the application still retrieves and prints evidence instead of pretending to have generated a grounded answer.

## Quickstart

### 1. Index your documents

Put research material in a directory such as:

    data/
    +-- papers/
    |   +-- paper-one.md
    |   +-- paper-two.md
    |   +-- notes.txt
    +-- sample/

Then run:

    research-assistant index data/papers

The generated local index is stored in data/index/ and is ignored by Git.

### 2. Ask questions

    research-assistant ask "What are the main advantages of retrieve-and-rerank?"

The index is loaded from disk, so the corpus is not re-embedded for every question.

### 3. Disable reranking

    research-assistant ask "What is RAG?" --no-reranker

This is useful when you want a faster retrieval-only baseline.

### 4. Change retrieval depth

    research-assistant ask "How does semantic retrieval work?" --top-k 12

### 5. Use a custom index directory

    research-assistant index data/papers --index-dir data/my-research
    research-assistant ask "Summarize the findings" --index-dir data/my-research

### 6. Index a web page

    research-assistant index-url https://example.com/research

You can also choose a custom index directory:

    research-assistant index-url https://example.com/research --index-dir data/web-index

## Included example

The repository contains a tiny sample corpus in data/sample/.

    research-assistant index data/sample
    research-assistant ask "Why use a bi-encoder before a cross-encoder?"

## Project structure

    ai-research-assistant/
    +-- data/
    |   +-- index/                  # Local persisted index
    |   +-- sample/                 # Small example corpus
    +-- research_assistant/
    |   +-- __init__.py
    |   +-- chunking.py             # Chunking and overlap
    |   +-- cli.py                  # Command-line interface
    |   +-- ingestion.py            # Document discovery/loading
    |   +-- llm.py                  # OpenAI-compatible LLM client
    |   +-- models.py               # Core data models
    |   +-- pipeline.py             # End-to-end orchestration
    |   +-- retrieval.py            # Dense, BM25, fusion, and reranking
    |   +-- storage.py              # Local index persistence
    |   +-- vector_store.py         # Vector-store protocol and Qdrant backend
    |   +-- web_search.py           # Web search provider abstraction and Tavily implementation
    +-- tests/
    |   +-- test_bm25.py
    |   +-- test_chunking.py
    |   +-- test_cli.py
    |   +-- test_fusion.py
    |   +-- test_ingestion.py
    |   +-- test_models.py
    |   +-- test_pipeline.py
    |   +-- test_retrieval.py
    |   +-- test_reranker.py
    |   +-- test_storage.py
    |   +-- test_vector_store.py
    |   +-- test_web_search.py
    +-- .env.example
    +-- .gitignore
    +-- pyproject.toml
    +-- README.md

## Design decisions

### Why hybrid retrieval?

Dense retrieval captures semantic similarity, while BM25 provides a strong lexical matching signal. Combining both result lists with reciprocal rank fusion improves retrieval robustness. Dense retrieval can use the lightweight local NumPy backend or Qdrant when a scalable vector-store architecture is needed.

### Why retrieve + rerank?

Dense and lexical retrieval efficiently produce a candidate set. A CrossEncoder then scores the candidates jointly with the query. This separates the broad retrieval stage from the more expensive precision stage.

### Why persist the index?

Embedding an entire corpus can be much more expensive than embedding one query. Persisting chunks and embeddings makes the tool reusable instead of rebuilding its index for every question.

The default local NumPy backend keeps the project lightweight, while Qdrant provides a persistent vector-store backend for larger corpora and future deployment scenarios.

### Why OpenAI-compatible?

The LLM layer targets the common chat-completions interface rather than hard-coding one provider. Compatible hosted APIs and local inference servers can therefore be used without changing the retrieval pipeline.

## Current limitations

This is an early-stage research/RAG project, not a production platform.

- Web search is currently exposed as a provider-level component and is not yet integrated into the end-to-end research workflow.
- There is no automated multi-page crawling workflow yet.
- Citations are source labels supplied to the LLM, not independently verified claims.
- There is no retrieval/answer evaluation harness yet.

These limitations are intentional extension points.

## Roadmap

### Retrieval
- [x] Hybrid BM25 + dense retrieval
- [x] Metadata filtering
- [x] Document deduplication
- [x] Configurable embedding/reranker models
- [x] Qdrant vector database backend

### Research workflow
- [x] PDF ingestion
- [x] HTML/web ingestion
- [ ] Tavily search integration into the research pipeline
- [ ] Query expansion and multi-query retrieval
- [ ] Source-level answer verification
- [ ] Research report generation with bibliography

### Evaluation
- [ ] Golden-question dataset
- [ ] Recall@K, MRR and NDCG
- [ ] Answer faithfulness checks
- [ ] Retrieval-vs-generation error analysis

### Developer experience
- [ ] CI with linting and tests
- [ ] Structured logging
- [ ] Typed configuration object
- [ ] Optional FastAPI service
- [ ] Web UI

## Development

Run the test suite:

    pytest

Install in editable mode:

    pip install -e ".[dev]"

## Contributing

Keep contributions focused and modular. Behavior changes should include tests where practical. Provider-specific code should remain isolated from the core retrieval pipeline.

## License

MIT