"""Weighted Reciprocal Rank Fusion.

Dense (cosine) and sparse (BM25) scores live on incompatible scales, so we fuse
by rank rather than raw score. Each list contributes weight / (rrf_k + rank) to a
chunk's fused score; weights are configurable so dense/sparse balance can be tuned
per corpus and the dashboard can A/B it. Doing fusion client-side (rather than via
Qdrant's server-side fusion) is what makes those weights adjustable.
"""

from __future__ import annotations

from typing import List, Tuple

from app.config import settings
from app.models.schemas import Chunk, RetrievedChunk


def reciprocal_rank_fusion(
    dense: List[Tuple[Chunk, float]],
    sparse: List[Tuple[Chunk, float]],
    *,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    rrf_k: int | None = None,
) -> List[RetrievedChunk]:
    dense_weight = settings.rrf_dense_weight if dense_weight is None else dense_weight
    sparse_weight = settings.rrf_sparse_weight if sparse_weight is None else sparse_weight
    rrf_k = settings.rrf_k if rrf_k is None else rrf_k

    merged: dict[str, RetrievedChunk] = {}

    def _entry(chunk: Chunk) -> RetrievedChunk:
        if chunk.id not in merged:
            merged[chunk.id] = RetrievedChunk(chunk=chunk, fused_score=0.0)
        return merged[chunk.id]

    for rank, (chunk, score) in enumerate(dense):
        entry = _entry(chunk)
        entry.dense_score = score
        entry.fused_score += dense_weight / (rrf_k + rank)

    for rank, (chunk, score) in enumerate(sparse):
        entry = _entry(chunk)
        entry.sparse_score = score
        entry.fused_score += sparse_weight / (rrf_k + rank)

    return sorted(merged.values(), key=lambda rc: rc.fused_score or 0.0, reverse=True)
