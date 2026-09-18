# AI Research Assistant

An end-to-end research assistant that ingests documents, retrieves relevant evidence, optionally re-ranks candidates, and produces a citation-aware answer through an OpenAI-compatible chat endpoint.

## Architecture

```text
Query
  │
  ├──► Dense retrieval (Sentence Transformers)
  │
  ├──► Optional cross-encoder re-ranking
  │
  └──► Evidence pack
          │
          ▼
   LLM answer synthesis
          │
          ▼
  Answer + source citations
```

The retrieval/re-ranking design follows the established two-stage pattern: a fast bi-encoder retrieves candidates and a slower cross-encoder improves their ordering.

## Features

- Clean Python package structure
- Markdown/text document ingestion
- Chunking with source metadata
- Dense semantic retrieval
- Optional CrossEncoder re-ranking
- Citation-ready evidence context
- OpenAI-compatible LLM endpoint
- CLI interface
- Unit tests
- `.env.example`
- No API keys committed to the repository

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Add your model endpoint/key to `.env` if you want generated answers.

Index sample documents:

```bash
research-assistant index data/sample
```

Ask a question:

```bash
research-assistant ask "What are the main advantages of retrieve-and-rerank?"
```

Run tests:

```bash
pytest
```

## Project status

Day 01 portfolio project — intentionally designed as a foundation that can be extended with web crawling, hybrid BM25+dense retrieval, evaluation datasets, observability, and a web UI.

## Why retrieve + rerank?

Semantic embeddings are efficient for finding a broad candidate set, while a cross-encoder can score query-document pairs more precisely. Using the two stages together gives a practical accuracy/latency trade-off.

## License

MIT
