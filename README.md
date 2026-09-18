# AI Research Assistant

An end-to-end research assistant that retrieves relevant evidence from a local document collection, optionally reranks candidates, and produces a citation-aware answer through an OpenAI-compatible chat endpoint.

## Architecture

```
Question
   │
   ├──► Dense semantic retrieval
   │
   ├──► Optional CrossEncoder reranking
   │
   └──► Evidence pack
            │
            ▼
       LLM synthesis
            │
            ▼
      Answer + citations
```

## Features

- Markdown and text ingestion
- Overlapping document chunking
- Dense semantic retrieval with Sentence Transformers
- Optional CrossEncoder reranking
- Source-aware evidence context
- OpenAI-compatible LLM endpoint
- CLI interface
- Unit tests
- Environment-based configuration
- No credentials committed to the repository

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env
```

The sample corpus is included in `data/sample`.

Index and query it:

```bash
research-assistant ask "Why is retrieve and rerank useful?"
```

Run tests:

```bash
pytest
```

## Retrieval strategy

The first stage uses a bi-encoder to efficiently retrieve a candidate set. The optional second stage uses a CrossEncoder to score query-document pairs more precisely. This two-stage retrieve-and-rerank pattern gives a practical accuracy/latency trade-off.

## Roadmap

- [ ] Persistent vector index
- [ ] Hybrid BM25 + dense retrieval
- [ ] Retrieval evaluation with Recall@k, MRR and NDCG
- [ ] Web/document ingestion
- [ ] Streaming answers
- [ ] FastAPI service
- [ ] Docker image
- [ ] CI pipeline
- [ ] Web UI

## License

MIT
