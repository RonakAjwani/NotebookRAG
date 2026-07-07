"""Core data shapes for the RAG pipeline.

`Chunk` is the ingestion unit (text + provenance metadata). `RetrievedChunk`
wraps a chunk with retrieval scores. The API response models sit at the bottom.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class Chunk(BaseModel):
    """A single indexed passage plus where it came from."""

    id: str = Field(..., description="Stable deterministic id (content+source hash)")
    text: str
    source: str = Field(..., description="Source document filename / id")
    doc_id: str = Field(..., description="Stable id of the parent document")
    chunk_index: int
    section: Optional[str] = Field(None, description="Nearest section heading")
    page: Optional[int] = Field(None, description="Page number for PDFs")
    strategy: ChunkStrategy
    char_count: int = 0

    def to_payload(self) -> dict[str, Any]:
        """Metadata payload stored alongside vectors in Qdrant."""
        return {
            "text": self.text,
            "source": self.source,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "page": self.page,
            "strategy": self.strategy.value,
            "char_count": self.char_count,
        }


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, carrying its scores through the pipeline."""

    chunk: Chunk
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Citation(BaseModel):
    marker: int = Field(..., description="The [n] index used in the answer text")
    doc_id: str
    source: str
    section: Optional[str] = None
    page: Optional[int] = None
    text: str = Field(..., description="The cited chunk content")
    verified: Optional[bool] = Field(
        None, description="Whether the judge confirmed this citation supports its claim"
    )


class ConfidenceBreakdown(BaseModel):
    retrieval: float = Field(..., ge=0.0, le=1.0)
    citation_coverage: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    composite: float = Field(..., ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    answered: bool = Field(..., description="False when the I-don't-know path triggered")
    mode: RetrievalMode
    citations: List[Citation] = Field(default_factory=list)
    confidence: Optional[ConfidenceBreakdown] = None
    retrieved: List[RetrievedChunk] = Field(default_factory=list)
    suggested_sources: List[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    chunk_count: int
    strategies: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    documents: List[DocumentInfo]
    chunks_indexed: int
    duplicates_skipped: int
