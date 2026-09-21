from pathlib import Path
from typing import Protocol, runtime_checkable
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams

from .models import DocumentChunk


@runtime_checkable
class VectorStoreProtocol(Protocol):
	def upsert(
		self,
		chunks: list[DocumentChunk],
		embeddings: np.ndarray,
	) -> None:
		...

	def search(
		self,
		query_embedding: np.ndarray,
		top_k: int,
		metadata_filter: dict[str, str] | None = None,
	) -> list[tuple[DocumentChunk, float]]:
		...

	def load_chunks(self) -> list[DocumentChunk]:
		...

	def close(self) -> None:
		...

class QdrantVectorStore:
	"""Qdrant-backed vector store for semantic retrieval."""

	collection_name = "research_chunks"

	def __init__(self, directory: str | Path):
		self.directory = Path(directory)
		self.client = QdrantClient(path=str(self.directory))

	def upsert(
		self,
		chunks: list[DocumentChunk],
		embeddings: np.ndarray,
	) -> None:
		if embeddings.ndim != 2:
			raise ValueError("embeddings must be a 2-dimensional array")

		if len(chunks) != len(embeddings):
			raise ValueError("chunks and embeddings must have the same length")

		if not chunks:
			return

		vector_size = embeddings.shape[1]

		if self.client.collection_exists(self.collection_name):
			collection_info = self.client.get_collection(self.collection_name)
			existing_size = collection_info.config.params.vectors.size

			if existing_size != vector_size:
				raise ValueError(
					"embedding dimension does not match existing collection"
				)

			self.client.delete(
				collection_name=self.collection_name,
				points_selector=Filter(),
			)
		else:
			self.client.create_collection(
				collection_name=self.collection_name,
				vectors_config=VectorParams(
					size=vector_size,
					distance=Distance.COSINE,
				),
			)

		points = []

		for chunk, embedding in zip(chunks, embeddings):
			point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

			points.append(
				PointStruct(
					id=point_id,
					vector=embedding.tolist(),
					payload={
						"chunk_id": chunk.chunk_id,
						"source": chunk.source,
						"text": chunk.text,
						"metadata": chunk.metadata,
					},
				)
			)

		self.client.upsert(
			collection_name=self.collection_name,
			points=points,
		)

	def close(self) -> None:
		self.client.close()

	def load_chunks(self) -> list[DocumentChunk]:
		if not self.client.collection_exists(self.collection_name):
			raise FileNotFoundError(
				f"No Qdrant collection found: {self.collection_name}"
			)

		chunks: list[DocumentChunk] = []
		offset = None

		while True:
			points, offset = self.client.scroll(
				collection_name=self.collection_name,
				limit=100,
				offset=offset,
				with_payload=True,
				with_vectors=False,
			)

			for point in points:
				payload = point.payload or {}

				chunks.append(
					DocumentChunk(
						chunk_id=payload["chunk_id"],
						source=payload["source"],
						text=payload["text"],
						metadata=payload.get("metadata", {}),
					)
				)

			if offset is None:
				break

		return chunks

	def search(
		self,
		query_embedding: np.ndarray,
		top_k: int,
		metadata_filter: dict[str, str] | None = None,
	) -> list[tuple[DocumentChunk, float]]:
		if query_embedding.ndim != 1:
			raise ValueError("query_embedding must be a 1-dimensional array")

		if top_k < 1:
			raise ValueError("top_k must be at least 1")

		query_filter = None

		if metadata_filter:
			from qdrant_client.models import FieldCondition, Filter, MatchValue

			query_filter = Filter(
				must=[
					FieldCondition(
						key=f"metadata.{key}",
						match=MatchValue(value=value),
					)
					for key, value in metadata_filter.items()
				]
			)

		results = self.client.query_points(
			collection_name=self.collection_name,
			query=query_embedding.tolist(),
			query_filter=query_filter,
			limit=top_k,
		).points

		return [
			(
				DocumentChunk(
					chunk_id=result.payload["chunk_id"],
					source=result.payload["source"],
					text=result.payload["text"],
					metadata=result.payload.get("metadata", {}),
				),
				result.score,
			)
			for result in results
		]