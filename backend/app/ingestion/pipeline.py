"""Ingestion orchestration: load -> persist raw+processed -> chunk -> embed ->
dedup -> index into Qdrant.

Raw and processed copies are kept so the corpus can be re-indexed (e.g. to try a
different chunking strategy) without re-uploading the originals.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Iterable, List

from app.config import settings
from app.ingestion.chunkers import chunk_document
from app.ingestion.dedup import Deduplicator
from app.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from app.models.schemas import ChunkStrategy, DocumentInfo, IngestResponse
from app.retrieval.embeddings import get_embedder
from app.retrieval.vector_store import get_store


def _iter_files(paths: Iterable[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in names:
                    if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
                        files.append(os.path.join(root, name))
        elif os.path.isfile(path):
            files.append(path)
    return sorted(files)


def _persist(path: str, doc) -> None:
    os.makedirs(settings.raw_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)

    raw_target = os.path.join(settings.raw_dir, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(raw_target):
        shutil.copy2(path, raw_target)

    processed = {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "blocks": [
            {"text": b.text, "section": b.section, "page": b.page} for b in doc.blocks
        ],
    }
    with open(os.path.join(settings.processed_dir, f"{doc.doc_id}.json"), "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)


def ingest_paths(
    paths: Iterable[str],
    strategy: ChunkStrategy | str | None = None,
    reset: bool = False,
) -> IngestResponse:
    store = get_store()
    if reset:
        store.reset()
    else:
        store.ensure_collection()

    embedder = get_embedder()
    dedup = Deduplicator(store)

    documents: List[DocumentInfo] = []
    total_indexed = 0
    total_skipped = 0

    for path in _iter_files(paths):
        doc = load_document(path)
        _persist(path, doc)

        chunks = chunk_document(doc, strategy)
        if not chunks:
            documents.append(DocumentInfo(doc_id=doc.doc_id, source=doc.source, chunk_count=0))
            continue

        texts = [c.text for c in chunks]
        dense_vecs = embedder.embed_documents_dense(texts)
        sparse_vecs = embedder.embed_documents_sparse(texts)

        kept_c, kept_d, kept_s, skipped = dedup.filter(chunks, dense_vecs, sparse_vecs)
        store.upsert_chunks(kept_c, kept_d, kept_s)

        total_indexed += len(kept_c)
        total_skipped += skipped
        documents.append(
            DocumentInfo(
                doc_id=doc.doc_id,
                source=doc.source,
                chunk_count=len(kept_c),
                strategies=sorted({c.strategy.value for c in kept_c}),
            )
        )

    return IngestResponse(
        documents=documents,
        chunks_indexed=total_indexed,
        duplicates_skipped=total_skipped,
    )
