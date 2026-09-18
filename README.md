# AI Research Assistant

A local-first, citation-aware research assistant for turning Markdown and text documents into searchable evidence and grounded LLM answers.

## What it does

The project implements a practical retrieve → rerank → synthesize pipeline:

1. Ingest Markdown/text documents and preserve their source paths.
2. Split documents into overlapping chunks.
3. Create dense embeddings with Sentence Transformers.
4. Persist chunks and embeddings to a local index.
5. Retrieve the most relevant candidates for a question.
6. Optionally rerank candidates with a CrossEncoder.
7. Send only the retrieved evidence to an OpenAI-compatible chat endpoint.
8. Ask the model to cite the supplied sources as [1], [2], etc.

This keeps the core small while leaving clear extension points for hybrid retrieval, web search, evaluation, PDF ingestion, and a UI.

## Architecture

    Documents
        │
        ▼
    Ingestion → Chunking + source metadata
        │
        ▼
    Sentence Transformer embeddings
        │
        ▼
    Persistent local index
        │
        ▼
    Dense retrieval → candidate chunks
        │
        ▼
    Optional CrossEncoder reranking
        │
        ▼
    Evidence context
        │
        ▼
    OpenAI-compatible LLM
        │
        ▼
    Answer + source citations

## Features

- Markdown and plain-text ingestion
- Overlapping chunking with source metadata
- Dense semantic retrieval
- Optional CrossEncoder reranking
- Persistent local NumPy-based index
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

The client expects the standard chat-completions route:

    POST {LLM_BASE_URL}/chat/completions

The LLM is optional. Without configuration, the application still retrieves and prints evidence instead of pretending to have generated a grounded answer.

## Quickstart

### 1. Index your documents

Put research material in a directory such as:

    data/
    ├── papers/
    │   ├── paper-one.md
    │   ├── paper-two.md
    │   └── notes.txt
    └── sample/

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

## Included example

The repository contains a tiny sample corpus in data/sample/.

    research-assistant index data/sample
    research-assistant ask "Why use a bi-encoder before a cross-encoder?"

## Project structure

    ai-research-assistant/
    ├── data/
    │   └── sample/                 # Small example corpus
    ├── research_assistant/
    │   ├── chunking.py             # Chunking and overlap
    │   ├── cli.py                  # Command-line interface
    │   ├── ingestion.py            # Document discovery/loading
    │   ├── llm.py                  # OpenAI-compatible LLM client
    │   ├── models.py               # Core data models
    │   ├── pipeline.py             # End-to-end orchestration
    │   ├── retrieval.py            # Dense retrieval + reranking
    │   └── storage.py              # Local index persistence
    ├── tests/
    │   ├── test_chunking.py
    │   └── test_ingestion.py
    ├── .env.example
    ├── .gitignore
    ├── pyproject.toml
    └── README.md

## Design decisions

### Why retrieve + rerank?

Dense retrieval is efficient because documents and queries are embedded independently. A CrossEncoder then scores a smaller candidate set jointly with the query. This separates the fast broad search stage from the more expensive precision stage.

### Why persist the index?

Embedding an entire corpus can be much more expensive than embedding one query. Persisting chunks and embeddings makes the tool reusable instead of rebuilding its index for every question.

### Why OpenAI-compatible?

The LLM layer targets the common chat-completions interface rather than hard-coding one provider. Compatible hosted APIs and local inference servers can therefore be used without changing the retrieval pipeline.

## Current limitations

This is an early-stage research/RAG project, not a production platform.

- Only .md and .txt files are ingested.
- The index is a local NumPy-based store.
- Retrieval is dense-only; BM25/hybrid retrieval is not implemented yet.
- Citations are source labels supplied to the LLM, not independently verified claims.
- There is no web search or crawling layer.
- There is no retrieval/answer evaluation harness yet.
- Very large corpora will eventually need a scalable vector index.

These limitations are intentional extension points.

## Roadmap

### Retrieval
- [ ] Hybrid BM25 + dense retrieval
- [ ] Metadata filtering
- [ ] Document deduplication
- [ ] Configurable embedding/reranker models
- [ ] Scalable vector database backend

### Research workflow
- [ ] PDF ingestion
- [ ] HTML/web ingestion
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
