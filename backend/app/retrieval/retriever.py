"""Retrieval orchestration.

Modes:
  - dense:  vector search only
  - sparse: BM25 only
  - hybrid: dense + sparse -> weighted RRF -> listwise LLM rerank

The mode switch exists so the dashboard can show hybrid vs dense-only side by
side and the eval harness can measure each in isolation. Reranking is applied on
the hybrid path (and optionally to dense/sparse via `rerank=`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.config import settings
from app.models.schemas import RetrievalMode, RetrievedChunk
from app.retrieval.embeddings import get_embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import RerankError, rerank as llm_rerank
from app.retrieval.vector_store import get_store


@dataclass
class RetrievalResult:
    """Retrieved chunks plus whether the LLM rerank actually ran.

    When the rerank judge fails we degrade to fused order but say so - callers
    (the generator, the eval runner) must be able to tell a reranked top-k from
    a fallback, or failure becomes indistinguishable from signal.
    """

    chunks: List[RetrievedChunk] = field(default_factory=list)
    rerank_ok: bool = True
    error: str | None = None


def retrieve(
    question: str,
    mode: RetrievalMode | str = RetrievalMode.HYBRID,
    *,
    top_k: int | None = None,
    rerank: bool | None = None,
) -> RetrievalResult:
    mode = RetrievalMode(mode)
    top_k = settings.rerank_top_k if top_k is None else top_k
    embedder = get_embedder()
    store = get_store()

    if mode is RetrievalMode.DENSE:
        hits = store.search_dense(embedder.embed_query_dense(question), settings.rerank_candidates)
        candidates = [RetrievedChunk(chunk=c, dense_score=s, fused_score=s) for c, s in hits]
        do_rerank = bool(rerank)

    elif mode is RetrievalMode.SPARSE:
        hits = store.search_sparse(embedder.embed_query_sparse(question), settings.rerank_candidates)
        candidates = [RetrievedChunk(chunk=c, sparse_score=s, fused_score=s) for c, s in hits]
        do_rerank = bool(rerank)

    else:  # hybrid
        dense_hits = store.search_dense(embedder.embed_query_dense(question), settings.dense_top_k)
        sparse_hits = store.search_sparse(embedder.embed_query_sparse(question), settings.sparse_top_k)
        candidates = reciprocal_rank_fusion(dense_hits, sparse_hits)[: settings.rerank_candidates]
        do_rerank = True if rerank is None else rerank

    if not candidates:
        return RetrievalResult(chunks=[])

    if do_rerank:
        try:
            return RetrievalResult(chunks=llm_rerank(question, candidates, top_k=top_k))
        except RerankError as exc:
            # Degrade to fused order, loudly: no synthetic rerank scores.
            return RetrievalResult(
                chunks=candidates[:top_k], rerank_ok=False, error=str(exc)
            )
    return RetrievalResult(chunks=candidates[:top_k])
