# Retrieve and Rerank

A practical semantic retrieval pipeline can use a bi-encoder to retrieve a relatively large candidate set and a cross-encoder to rerank those candidates. The first stage is efficient because documents and queries are embedded independently. The second stage is more expensive because the query and candidate document are scored together.

This architecture is useful for semantic search and question answering because retrieval speed and ranking quality have different requirements.
