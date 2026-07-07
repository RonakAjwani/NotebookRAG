"""Dense + sparse embeddings via fastembed (local, no API key).

Dense: BAAI/bge-small-en-v1.5 (384-dim). Sparse: Qdrant/bm25 (IDF-weighted).
Models are ONNX and loaded lazily on first use so importing this module is
cheap and side-effect-free. BM25 distinguishes document vs query embedding, so
we expose separate document/query methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from app.config import settings


@dataclass
class SparseVector:
    indices: List[int]
    values: List[float]


class Embedder:
    def __init__(self, dense_model: str | None = None, sparse_model: str | None = None):
        self.dense_model_name = dense_model or settings.dense_model
        self.sparse_model_name = sparse_model or settings.sparse_model
        self._dense = None
        self._sparse = None

    # -- lazy loaders -------------------------------------------------------

    @property
    def dense(self):
        if self._dense is None:
            from fastembed import TextEmbedding

            self._dense = TextEmbedding(model_name=self.dense_model_name)
        return self._dense

    @property
    def sparse(self):
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse

    # -- dense --------------------------------------------------------------

    def embed_documents_dense(self, texts: Iterable[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self.dense.embed(list(texts))]

    def embed_query_dense(self, text: str) -> List[float]:
        # bge models support a dedicated query embedding path.
        try:
            return next(iter(self.dense.query_embed(text))).tolist()
        except AttributeError:
            return next(iter(self.dense.embed([text]))).tolist()

    # -- sparse -------------------------------------------------------------

    def embed_documents_sparse(self, texts: Iterable[str]) -> List[SparseVector]:
        out = []
        for emb in self.sparse.embed(list(texts)):
            out.append(SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist()))
        return out

    def embed_query_sparse(self, text: str) -> SparseVector:
        emb = next(iter(self.sparse.query_embed(text)))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
