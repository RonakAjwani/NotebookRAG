"""Listwise LLM reranker (RankGPT-style).

One judge call scores all fused candidates for relevance to the question and
returns a ranked order. Listwise (not one-call-per-candidate) keeps it to a
single request per query - essential under free-tier rate limits - and temp=0
keeps ordering reproducible so eval comparisons are stable. The 0..1 relevance
scores feed the confidence scorer's retrieval dimension.

Failure is LOUD: if the judge call fails or returns unparseable rankings we
raise RerankError rather than silently falling back - a silent fallback made an
entire eval run's numbers indistinguishable from real quality signals once the
judge provider's quota was exhausted. Callers decide how to degrade.
"""

from __future__ import annotations

from typing import List

from app.config import settings
from app.llm.client import LLMError, get_rerank_client
from app.models.schemas import RetrievedChunk


class RerankError(RuntimeError):
    """The rerank judge call failed or returned nothing usable."""

_SYSTEM = (
    "You are a precise relevance judge for a retrieval system. You score how well "
    "each passage answers a question, independently, on a 0.0 to 1.0 scale."
)

_INSTRUCTION = """Question: {question}

Passages:
{passages}

Score each passage for how directly it helps answer the question:
- 1.0 = directly and fully answers it
- 0.5 = related / partial
- 0.0 = irrelevant

Output ONE line of compact, valid JSON and nothing else — no prose, no code
fences, no trailing commas. Exact shape:
{{"rankings":[{{"index":<passage number>,"relevance":<0.0-1.0>}}]}}
Include every passage exactly once."""


def _format_passages(candidates: List[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(candidates, start=1):
        header = rc.chunk.source
        if rc.chunk.section:
            header += f" — {rc.chunk.section}"
        snippet = rc.chunk.text[:450]
        lines.append(f"[{i}] ({header})\n{snippet}")
    return "\n\n".join(lines)


def rerank(question: str, candidates: List[RetrievedChunk], top_k: int | None = None) -> List[RetrievedChunk]:
    top_k = settings.rerank_top_k if top_k is None else top_k
    if not candidates:
        return []
    if len(candidates) == 1:
        candidates[0].rerank_score = candidates[0].rerank_score or 1.0
        return candidates[:top_k]

    prompt = _INSTRUCTION.format(question=question, passages=_format_passages(candidates))
    try:
        # Generous max_tokens so a long ranking list never truncates mid-JSON.
        data = get_rerank_client().complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1200,
        )
        rankings = data.get("rankings", []) if isinstance(data, dict) else []
    except (LLMError, AttributeError) as exc:
        raise RerankError(f"rerank judge call failed: {exc}") from exc

    if not rankings:
        raise RerankError("rerank judge returned no parseable rankings")

    for item in rankings:
        try:
            idx = int(item["index"]) - 1
            rel = float(item["relevance"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= idx < len(candidates):
            candidates[idx].rerank_score = max(0.0, min(1.0, rel))

    for rc in candidates:
        if rc.rerank_score is None:
            rc.rerank_score = 0.0

    ranked = sorted(candidates, key=lambda rc: rc.rerank_score or 0.0, reverse=True)
    return ranked[:top_k]
