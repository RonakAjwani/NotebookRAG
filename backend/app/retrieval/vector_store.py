"""Qdrant wrapper: one collection, named dense + sparse vectors per point.

Storing both vectors on the same point makes the dense and sparse indexes
sync-by-construction: one upsert writes both, one delete removes both. Point ids
are UUID5-derived from the chunk's content id (Qdrant only accepts int/UUID ids),
with the original id kept in the payload.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from app.config import settings
from app.models.schemas import Chunk
from app.retrieval.embeddings import SparseVector

_NAMESPACE = uuid.UUID("6f3c9e2a-1b7d-4e5a-9c8b-0d1e2f3a4b5c")

DENSE = "dense"
SPARSE = "sparse"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore:
    def __init__(self, collection: str | None = None):
        from qdrant_client import QdrantClient

        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )

    # -- schema -------------------------------------------------------------

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=settings.dense_dimension, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        # Index doc_id so document-scoped filtering / deletion is fast.
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="doc_id",
            field_schema="keyword",
        )

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.ensure_collection()

    # -- writes -------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        dense_vecs: List[List[float]],
        sparse_vecs: List[SparseVector],
    ) -> None:
        from qdrant_client import models

        points = []
        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs):
            payload = chunk.to_payload()
            payload["chunk_id"] = chunk.id
            points.append(
                models.PointStruct(
                    id=_point_id(chunk.id),
                    vector={
                        DENSE: dvec,
                        SPARSE: models.SparseVector(indices=svec.indices, values=svec.values),
                    },
                    payload=payload,
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def delete_document(self, doc_id: str) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    # -- reads --------------------------------------------------------------

    def search_dense(self, vector: List[float], k: int) -> List[Tuple[Chunk, float]]:
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE,
            limit=k,
            with_payload=True,
        )
        return [(_chunk_from_payload(p.payload), p.score) for p in resp.points]

    def search_sparse(self, sparse: SparseVector, k: int) -> List[Tuple[Chunk, float]]:
        from qdrant_client import models

        resp = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=SPARSE,
            limit=k,
            with_payload=True,
        )
        return [(_chunk_from_payload(p.payload), p.score) for p in resp.points]

    def nearest_dense_score(self, vector: List[float]) -> Optional[float]:
        """Top-1 cosine score against what's already indexed (for dedup)."""
        resp = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE,
            limit=1,
            with_payload=False,
        )
        return resp.points[0].score if resp.points else None

    def list_documents(self) -> List[dict]:
        """Aggregate indexed points into per-document summaries."""
        docs: dict[str, dict] = {}
        next_page = None
        while True:
            points, next_page = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("doc_id", "unknown")
                entry = docs.setdefault(
                    doc_id,
                    {"doc_id": doc_id, "source": payload.get("source", ""), "chunk_count": 0, "strategies": set()},
                )
                entry["chunk_count"] += 1
                if payload.get("strategy"):
                    entry["strategies"].add(payload["strategy"])
            if next_page is None:
                break
        for entry in docs.values():
            entry["strategies"] = sorted(entry["strategies"])
        return list(docs.values())

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count


def _chunk_from_payload(payload: dict) -> Chunk:
    return Chunk(
        id=payload.get("chunk_id", ""),
        text=payload.get("text", ""),
        source=payload.get("source", ""),
        doc_id=payload.get("doc_id", ""),
        chunk_index=payload.get("chunk_index", 0),
        section=payload.get("section"),
        page=payload.get("page"),
        strategy=payload.get("strategy", "recursive"),
        char_count=payload.get("char_count", 0),
    )


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
