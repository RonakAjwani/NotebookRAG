"""Three switchable chunking strategies.

- fixed:     baseline. Concatenate blocks, slide a fixed-size char window with
             overlap. Structure-agnostic.
- recursive: structure-aware. Split within each block (blocks are already cut on
             headings) via LangChain's RecursiveCharacterTextSplitter, preserving
             section/page.
- semantic:  topic-aware. Split blocks into sentences, embed them, and start a new
             chunk where consecutive-sentence cosine similarity drops below a
             threshold.

Every chunk records which strategy produced it, so the eval harness can compare
strategies on identical corpora. Chunk ids are deterministic (idempotent upserts).
"""

from __future__ import annotations

import hashlib
import re
from typing import List

from app.config import settings
from app.ingestion.loaders import Block, LoadedDocument
from app.models.schemas import Chunk, ChunkStrategy


def _chunk_id(doc_id: str, strategy: str, index: int, text: str) -> str:
    h = hashlib.sha256(f"{doc_id}|{strategy}|{index}|{text}".encode("utf-8")).hexdigest()
    return h[:24]


def _make_chunk(doc: LoadedDocument, strategy: ChunkStrategy, index: int, block: Block, text: str) -> Chunk:
    text = text.strip()
    return Chunk(
        id=_chunk_id(doc.doc_id, strategy.value, index, text),
        text=text,
        source=doc.source,
        doc_id=doc.doc_id,
        chunk_index=index,
        section=block.section,
        page=block.page,
        strategy=strategy,
        char_count=len(text),
    )


# -- fixed ------------------------------------------------------------------


def chunk_fixed(doc: LoadedDocument, size: int, overlap: int) -> List[Chunk]:
    # Build combined text with an offset -> owning block map for provenance.
    pieces: List[str] = []
    owners: List[Block] = []
    cursor = 0
    offsets: list[tuple[int, Block]] = []
    for block in doc.blocks:
        offsets.append((cursor, block))
        pieces.append(block.text)
        cursor += len(block.text) + 2  # matches the "\n\n" join below
    combined = "\n\n".join(pieces)

    def owner_at(pos: int) -> Block:
        owner = doc.blocks[0] if doc.blocks else Block(text="")
        for start, block in offsets:
            if start <= pos:
                owner = block
            else:
                break
        return owner

    chunks: List[Chunk] = []
    step = max(1, size - overlap)
    idx = 0
    for start in range(0, max(1, len(combined)), step):
        window = combined[start : start + size]
        if not window.strip():
            continue
        chunks.append(_make_chunk(doc, ChunkStrategy.FIXED, idx, owner_at(start), window))
        idx += 1
        if start + size >= len(combined):
            break
    return chunks


# -- recursive --------------------------------------------------------------


def chunk_recursive(doc: LoadedDocument, size: int, overlap: int) -> List[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: List[Chunk] = []
    idx = 0
    for block in doc.blocks:
        for piece in splitter.split_text(block.text):
            if piece.strip():
                chunks.append(_make_chunk(doc, ChunkStrategy.RECURSIVE, idx, block, piece))
                idx += 1
    return chunks


# -- semantic ---------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _cosine(a, b) -> float:
    import numpy as np

    va, vb = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
    return float(np.dot(va, vb) / denom)


def chunk_semantic(doc: LoadedDocument, threshold: float, max_size: int) -> List[Chunk]:
    from app.retrieval.embeddings import get_embedder

    embedder = get_embedder()
    chunks: List[Chunk] = []
    idx = 0
    for block in doc.blocks:
        sentences = _sentences(block.text)
        if len(sentences) <= 1:
            if block.text.strip():
                chunks.append(_make_chunk(doc, ChunkStrategy.SEMANTIC, idx, block, block.text))
                idx += 1
            continue

        vectors = embedder.embed_documents_dense(sentences)
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            sim = _cosine(vectors[i - 1], vectors[i])
            projected = len(" ".join(current)) + len(sentences[i])
            if sim < threshold or projected > max_size:
                chunks.append(_make_chunk(doc, ChunkStrategy.SEMANTIC, idx, block, " ".join(current)))
                idx += 1
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        if current:
            chunks.append(_make_chunk(doc, ChunkStrategy.SEMANTIC, idx, block, " ".join(current)))
            idx += 1
    return chunks


def chunk_document(doc: LoadedDocument, strategy: ChunkStrategy | str | None = None) -> List[Chunk]:
    strategy = ChunkStrategy(strategy) if strategy else ChunkStrategy(settings.chunk_strategy)
    if strategy is ChunkStrategy.FIXED:
        return chunk_fixed(doc, settings.chunk_size, settings.chunk_overlap)
    if strategy is ChunkStrategy.RECURSIVE:
        return chunk_recursive(doc, settings.chunk_size, settings.chunk_overlap)
    return chunk_semantic(doc, settings.semantic_threshold, settings.chunk_size)
