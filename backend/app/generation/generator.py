"""End-to-end answer pipeline: retrieve -> gate -> generate -> verify -> score.

The "I don't know" gate runs before generation: if mean retrieval relevance is
below threshold, we return a structured found/not-found response naming the docs
worth checking manually instead of letting the LLM fabricate.
"""

from __future__ import annotations

from typing import List

from app.config import settings
from app.generation.confidence import retrieval_confidence, score_confidence
from app.generation.prompts import build_generation_messages
from app.generation.verification import (
    VerificationError,
    extract_citations,
    normalize_citation_markers,
    verify,
)
from app.llm.client import get_generator
from app.models.schemas import (
    AnswerResponse,
    ConfidenceBreakdown,
    RetrievalMode,
    RetrievedChunk,
)
from app.retrieval.retriever import retrieve


def _suggested_sources(chunks: List[RetrievedChunk]) -> List[str]:
    seen: List[str] = []
    for rc in chunks:
        if rc.chunk.source not in seen:
            seen.append(rc.chunk.source)
    return seen[:3]


def _idk_response(question: str, mode: RetrievalMode, chunks: List[RetrievedChunk], retrieval: float) -> AnswerResponse:
    suggestions = _suggested_sources(chunks)
    if suggestions:
        msg = (
            "I couldn't find enough relevant information in the indexed documents to "
            "answer this confidently. The closest material appears in: "
            + ", ".join(suggestions)
            + " - these may be worth checking manually."
        )
    else:
        msg = (
            "I couldn't find any relevant information in the indexed documents to "
            "answer this question."
        )
    return AnswerResponse(
        question=question,
        answer=msg,
        answered=False,
        mode=mode,
        citations=[],
        confidence=ConfidenceBreakdown(
            retrieval=round(retrieval, 4), citation_coverage=0.0, completeness=0.0, composite=round(retrieval * 0.4, 4)
        ),
        retrieved=chunks,
        suggested_sources=suggestions,
        metadata={"reason": "below_retrieval_confidence_threshold"},
    )


def answer_question(
    question: str,
    mode: RetrievalMode | str = RetrievalMode.HYBRID,
    *,
    verify_citations: bool = True,
) -> AnswerResponse:
    mode = RetrievalMode(mode)
    errors: list[str] = []

    result = retrieve(question, mode)
    chunks = result.chunks
    if not result.rerank_ok:
        errors.append("rerank_failed")

    if not chunks:
        return _idk_response(question, mode, [], 0.0)

    # "I don't know" gate. Only meaningful when the rerank actually ran - a
    # fused-order fallback has no calibrated relevance scores to gate on.
    retrieval = retrieval_confidence(chunks)
    if result.rerank_ok and retrieval < settings.retrieval_confidence_threshold:
        return _idk_response(question, mode, chunks, retrieval)

    # Generate grounded answer. Normalize citation brackets to canonical [n] so
    # every downstream consumer (extraction, verification, frontend) agrees.
    messages = build_generation_messages(question, chunks)
    answer_text = normalize_citation_markers(get_generator().complete(messages))

    # Verify citations (always on) and score confidence. A dead verification
    # judge is recorded as an error, never scored as zero coverage.
    citations = extract_citations(answer_text, chunks)
    coverage = 0.0
    if verify_citations and citations:
        try:
            citations, coverage = verify(answer_text, chunks, citations)
        except VerificationError:
            errors.append("verification_failed")
            for citation in citations:
                citation.verified = None  # unknown, not unsupported

    confidence, conf_errors = score_confidence(question, answer_text, chunks, coverage)
    errors.extend(conf_errors)

    return AnswerResponse(
        question=question,
        answer=answer_text,
        answered=True,
        mode=mode,
        citations=citations,
        confidence=confidence,
        retrieved=chunks,
        metadata={
            "chunks_used": len(chunks),
            "generator_model": get_generator().model,
            "errors": errors,
        },
    )
