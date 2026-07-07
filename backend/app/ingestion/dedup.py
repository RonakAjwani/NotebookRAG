"""Two-stage deduplication.

1. Exact: sha256 of normalized text catches identical chunks for free (same
   passage copied across docs) without any vector math.
2. Near-duplicate: cosine similarity of the dense vector against what's already
   indexed; above `dedup_cosine_threshold` the chunk is skipped so the retriever
   doesn't spend context-window slots on redundant content.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Set, Tuple

from app.config import settings
from app.models.schemas import Chunk
from app.retrieval.embeddings import SparseVector


def _norm_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Deduplicator:
    def __init__(self, store, threshold: float | None = None):
        self.store = store
        self.threshold = threshold if threshold is not None else settings.dedup_cosine_threshold
        self._seen_hashes: Set[str] = set()

    def filter(
        self,
        chunks: List[Chunk],
        dense_vecs: List[List[float]],
        sparse_vecs: List[SparseVector],
    ) -> Tuple[List[Chunk], List[List[float]], List[SparseVector], int]:
        """Return the kept (chunks, dense, sparse) plus the count skipped."""
        keep_c, keep_d, keep_s = [], [], []
        skipped = 0
        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs):
            h = _norm_hash(chunk.text)
            if h in self._seen_hashes:
                skipped += 1
                continue

            top = self.store.nearest_dense_score(dvec)
            if top is not None and top >= self.threshold:
                skipped += 1
                self._seen_hashes.add(h)
                continue

            self._seen_hashes.add(h)
            keep_c.append(chunk)
            keep_d.append(dvec)
            keep_s.append(svec)
        return keep_c, keep_d, keep_s, skipped
