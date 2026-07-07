"""Prompt construction for grounded generation.

The context is rendered as numbered blocks [1..n] matching the reranked chunk
order, and the model is instructed to cite those markers and to refuse when the
context is insufficient (rather than fabricate).
"""

from __future__ import annotations

from typing import List

from app.models.schemas import RetrievedChunk

GENERATION_SYSTEM = """You are a precise assistant answering questions about internal documentation.

Rules:
1. Answer ONLY using the numbered context blocks provided. Do not use outside knowledge.
2. Cite the specific block(s) supporting each claim with bracketed markers like [1] or [2][3], placed inline right after the claim.
3. If the context does not contain enough information to answer, say so explicitly and do not guess.
4. Be concise and factual. Do not repeat the question.
"""


def build_context(chunks: List[RetrievedChunk]) -> str:
    blocks = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        loc = c.source
        if c.section:
            loc += f" — {c.section}"
        if c.page is not None:
            loc += f" (p.{c.page})"
        blocks.append(f"[{i}] Source: {loc}\n{c.text}")
    return "\n\n".join(blocks)


def build_generation_messages(question: str, chunks: List[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user = f"Context blocks:\n\n{context}\n\nQuestion: {question}\n\nAnswer with inline [n] citations:"
    return [
        {"role": "system", "content": GENERATION_SYSTEM},
        {"role": "user", "content": user},
    ]
