"""Citation verification.

Always-on (not optional): the confidence scorer's citation-coverage dimension
depends on it. Cost is controlled by per-answer batching - a single judge call
checks every (claim, cited-chunk) pair in one answer and returns a verdict per
pair. We never issue one call per claim, and never batch across answers (that
would blow context and degrade judge quality).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.llm.client import LLMError, get_verify_client
from app.models.schemas import Citation, RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

# Models don't always cite with ASCII brackets. gpt-oss-120b, for instance,
# emits 【1】 (CJK lenticular brackets); others use fullwidth ［1］ or 〔1〕.
# Normalize every variant to [n] so extraction, verification, and the frontend's
# clickable-citation parsing all see one canonical form. (Missing this alone
# scored correctly-cited answers as 0.0 citation accuracy in the first run.)
_MARKER_VARIANTS = re.compile(
    r"[\[［【〔❨❪⟦]\s*(\d+)\s*[\]］】〕❩❫⟧]"
)


def normalize_citation_markers(text: str) -> str:
    return _MARKER_VARIANTS.sub(r"[\1]", text)


class VerificationError(RuntimeError):
    """The verification judge call failed - coverage is unknown, not zero."""


_SYSTEM = (
    "You check whether a source passage supports a claim. A source supports a claim "
    "if it states or directly implies the claim's information — exact wording is not "
    "required, and a source may support a claim even if it also contains other content."
)

_INSTRUCTION = """For each item, decide whether the SOURCE passage supports the CLAIM.
Mark supported=true if the source states or directly implies what the claim asserts
(paraphrases and partial-but-on-point matches count). Mark supported=false only if
the source does not actually contain that information.

Items:
{items}

Return ONLY JSON:
{{"verdicts": [{{"id": <item id>, "supported": <true|false>}}, ...]}}
Include every item id exactly once."""


@dataclass
class _ClaimItem:
    claim_idx: int
    claim: str
    markers: List[int]
    combined_source: str


def extract_citations(answer: str, chunks: List[RetrievedChunk]) -> List[Citation]:
    """Build Citation objects for the markers actually used in the answer."""
    used = sorted({int(m) for m in _MARKER_RE.findall(answer)})
    citations: List[Citation] = []
    for marker in used:
        if 1 <= marker <= len(chunks):
            c = chunks[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    doc_id=c.doc_id,
                    source=c.source,
                    section=c.section,
                    page=c.page,
                    text=c.text,
                )
            )
    return citations


def _claims_with_markers(answer: str) -> List[Tuple[str, List[int]]]:
    """Split an answer into claim units with their citation markers.

    Line-aware, not just sentence-aware: models frequently answer with bullet
    lists whose items carry no terminal punctuation, so a pure sentence split
    mis-pairs claims and markers (this artifact alone dragged measured citation
    accuracy on lookup questions to 0.29 in the first eval run). Bullets become
    their own claims; a trailing line that is *only* markers (e.g. a lone "[1]"
    after a list) attaches to every marker-less claim before it.
    """
    out: List[Tuple[str, List[int]]] = []
    for raw_line in answer.strip().splitlines():
        line = _BULLET_RE.sub("", raw_line.strip())
        if not line:
            continue
        # A markers-only line ("[1]" or "[1][3]") cites the preceding claims.
        if _MARKER_RE.sub("", line).strip() in ("", ","):
            trailing = [int(m) for m in _MARKER_RE.findall(line)]
            for i, (claim, markers) in enumerate(out):
                if not markers:
                    out[i] = (claim, list(trailing))
            continue
        for sentence in _SENTENCE_RE.split(line):
            sentence = sentence.strip()
            if not sentence:
                continue
            markers = [int(m) for m in _MARKER_RE.findall(sentence)]
            out.append((sentence, markers))
    return out


def verify(
    answer: str,
    chunks: List[RetrievedChunk],
    citations: List[Citation],
) -> Tuple[List[Citation], float]:
    """Verify citations and return (citations-with-verified-flags, coverage).

    Verification is CLAIM-level against the UNION of a claim's cited sources -
    not one marker at a time. A sentence often cites [1][2] because different
    parts of it come from different chunks; judging each marker against the whole
    sentence in isolation wrongly fails both. So each claim is checked once
    against all its cited chunks combined, and every marker in a supported claim
    is marked verified.

    coverage = fraction of factual claims whose cited sources collectively
    support them.
    """
    claims = _claims_with_markers(answer)
    if not claims:
        return citations, 0.0

    # One verification item per claim that carries valid citation markers.
    items: List[_ClaimItem] = []
    for idx, (claim_text, markers) in enumerate(claims):
        valid = [m for m in markers if 1 <= m <= len(chunks)]
        if not valid:
            continue
        combined = "\n---\n".join(chunks[m - 1].chunk.text[:400] for m in valid)
        items.append(_ClaimItem(idx, claim_text, valid, combined))

    verdicts: Dict[int, bool] = {}
    if items:
        rendered = "\n\n".join(
            f"id {it.claim_idx}:\nCLAIM: {it.claim}\nSOURCE: {it.combined_source}" for it in items
        )
        try:
            data = get_verify_client().complete_json(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _INSTRUCTION.format(items=rendered)},
                ],
                temperature=0.0,
            )
            for v in data.get("verdicts", []) if isinstance(data, dict) else []:
                try:
                    verdicts[int(v["id"])] = bool(v["supported"])
                except (KeyError, ValueError, TypeError):
                    continue
        except LLMError as exc:
            # Unknown is not zero: a dead judge must not masquerade as
            # "all citations unsupported".
            raise VerificationError(f"verification judge call failed: {exc}") from exc
        if not verdicts:
            raise VerificationError("verification judge returned no parseable verdicts")

    # A marker is verified if any claim it appears in was collectively supported.
    marker_supported: Dict[int, bool] = {}
    for it in items:
        supported = verdicts.get(it.claim_idx, False)
        for m in it.markers:
            marker_supported[m] = marker_supported.get(m, False) or supported
    for citation in citations:
        citation.verified = marker_supported.get(citation.marker)

    covered = sum(1 for it in items if verdicts.get(it.claim_idx, False))
    coverage = covered / len(claims)
    return citations, coverage
