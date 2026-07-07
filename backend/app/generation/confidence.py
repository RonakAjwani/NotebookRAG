"""Answer confidence scoring across three dimensions.

- retrieval:        mean rerank relevance of the chunks used (how good was the
                    evidence?).
- citation_coverage: fraction of claims with a verified citation (from the
                    verification step).
- completeness: LLM-as-judge - did the answer address every part of the
                    question?

Composite is a weighted blend, returned with every answer so the caller (and the
"I don't know" gate) can reason about trust.
"""

from __future__ import annotations

from statistics import mean
from typing import List

from app.llm.client import LLMError, get_completeness_client
from app.models.schemas import ConfidenceBreakdown, RetrievedChunk

# Composite weights; retrieval is weighted highest since bad evidence dominates.
_W_RETRIEVAL = 0.4
_W_COVERAGE = 0.35
_W_COMPLETENESS = 0.25

_SYSTEM = "You assess whether an answer fully addresses a question."
_INSTRUCTION = """Question: {question}

Answer: {answer}

On a 0.0 to 1.0 scale, how completely does the answer address every part of the
question? 1.0 = fully addressed; 0.5 = partially; 0.0 = does not address it.

Return ONLY JSON: {{"completeness": <0.0-1.0>}}"""


def retrieval_confidence(chunks: List[RetrievedChunk]) -> float:
    """Confidence that the retrieved set contains usable evidence.

    Top-weighted rather than a flat mean: a precise lookup is often answered by a
    single strongly-relevant chunk while the other top-k are only tangential, so a
    flat mean would wrongly trip the "I don't know" gate. We blend the best hit
    with the average (0.6*max + 0.4*mean) - strong best evidence carries the gate,
    while the mean still rewards corroboration and penalizes an all-weak set.
    """
    scores = [rc.rerank_score for rc in chunks if rc.rerank_score is not None]
    if not scores:
        # No rerank scores (dense/sparse without rerank): fall back to fused,
        # normalized to 0..1 since fused/cosine scores are unbounded across modes.
        scores = [rc.fused_score for rc in chunks if rc.fused_score is not None]
        if not scores:
            return 0.0
        top = max(scores) or 1e-9
        scores = [s / top for s in scores]
    blended = 0.6 * max(scores) + 0.4 * mean(scores)
    return max(0.0, min(1.0, blended))


def completeness_score(question: str, answer: str) -> float | None:
    """0..1 completeness, or None if the judge call failed (unknown != bad)."""
    try:
        data = get_completeness_client().complete_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _INSTRUCTION.format(question=question, answer=answer)},
            ],
            temperature=0.0,
        )
        return max(0.0, min(1.0, float(data.get("completeness", 0.0))))
    except (LLMError, ValueError, TypeError, AttributeError):
        return None


def score_confidence(
    question: str,
    answer: str,
    chunks: List[RetrievedChunk],
    coverage: float,
) -> tuple[ConfidenceBreakdown, list[str]]:
    """Returns (breakdown, errors). A failed completeness judge uses a neutral
    0.5 prior in the composite but is reported in errors so callers (and the
    eval runner) can tell a measured score from a degraded one."""
    errors: list[str] = []
    retrieval = retrieval_confidence(chunks)
    completeness = completeness_score(question, answer)
    if completeness is None:
        errors.append("completeness_judge_failed")
        completeness = 0.5
    composite = (
        _W_RETRIEVAL * retrieval
        + _W_COVERAGE * coverage
        + _W_COMPLETENESS * completeness
    )
    return (
        ConfidenceBreakdown(
            retrieval=round(retrieval, 4),
            citation_coverage=round(coverage, 4),
            completeness=round(completeness, 4),
            composite=round(composite, 4),
        ),
        errors,
    )
