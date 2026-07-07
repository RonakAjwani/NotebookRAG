"""LLM-as-judge eval metrics.

Judged on the eval-judge provider (kept distinct from the generator to reduce
self-preference bias; can also differ from the pipeline judge to spread quota):
  - correctness + faithfulness in ONE combined call per case - halves judge
    load vs separate calls without changing what is measured.
Non-LLM metrics:
  - retrieval_hit:     did the expected source docs appear in the retrieved set?
  - citation_accuracy: fraction of the answer's citations that verified True.

Failure semantics: a failed judge call returns None - never 0.0. The first eval
run proved why: a quota-exhausted judge silently scored four whole configs as
0.0 across the board, indistinguishable from genuinely bad answers.
"""

from __future__ import annotations

from typing import Optional, Tuple

from app.evals.schemas import GoldenItem, QuestionCategory
from app.llm.client import LLMError, get_eval_judge
from app.models.schemas import AnswerResponse

_SYSTEM = "You are a strict evaluator of question-answering systems. Output JSON only."

_COMBINED = """Question: {question}

Reference answer: {reference}

Retrieved context (what the system had available):
{context}

System answer: {answer}

Score two independent dimensions, each 0.0-1.0:
1. "correctness": does the system answer match the reference in meaning?
   (1.0 fully correct, 0.5 partially, 0.0 wrong or missing)
2. "faithfulness": are ALL factual claims in the system answer supported by the
   retrieved context above? (1.0 fully grounded, 0.0 contains unsupported claims)

Return ONLY JSON: {{"correctness": <0.0-1.0>, "faithfulness": <0.0-1.0>}}"""


def score_answer(
    item: GoldenItem, answer: AnswerResponse
) -> Tuple[Optional[float], Optional[float]]:
    """(correctness, faithfulness) - None means the judge failed, not zero."""
    # no_answer items are scored objectively: did the system correctly abstain?
    if item.category is QuestionCategory.NO_ANSWER:
        if not answer.answered:
            return 1.0, 1.0
        # Wrongly answered: correctness is 0 by definition; still judge whether
        # the fabricated answer at least stayed grounded in what was retrieved.
        faith = _judged_faithfulness(item, answer) if answer.retrieved else None
        return 0.0, faith

    # An abstention on an answerable question is wrong but perfectly grounded.
    if not answer.answered:
        return 0.0, 1.0

    context = "\n\n".join(
        f"[{i+1}] {rc.chunk.text[:500]}" for i, rc in enumerate(answer.retrieved)
    ) or "(nothing retrieved)"
    prompt = _COMBINED.format(
        question=item.question,
        reference=item.answer or "(no reference: this question is unanswerable from the corpus)",
        context=context,
        answer=answer.answer,
    )
    try:
        data = get_eval_judge().complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        correctness = _clip(data.get("correctness"))
        faithfulness = _clip(data.get("faithfulness"))
        return correctness, faithfulness
    except (LLMError, ValueError, TypeError, AttributeError):
        return None, None


def _judged_faithfulness(item: GoldenItem, answer: AnswerResponse) -> Optional[float]:
    """Faithfulness for a no_answer question the system wrongly answered."""
    try:
        context = "\n\n".join(
            f"[{i+1}] {rc.chunk.text[:500]}" for i, rc in enumerate(answer.retrieved)
        )
        data = get_eval_judge().complete_json(
            [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context}\n\nAnswer: {answer.answer}\n\n"
                        'Are ALL factual claims in the answer supported by the context? '
                        'Return ONLY JSON: {"faithfulness": <0.0-1.0>}'
                    ),
                },
            ],
            temperature=0.0,
        )
        faith = _clip(data.get("faithfulness"))
    except (LLMError, ValueError, TypeError, AttributeError):
        faith = None
    return faith


def _clip(value) -> Optional[float]:
    try:
        return max(0.0, min(1.0, float(value)))
    except (ValueError, TypeError):
        return None


def score_retrieval_hit(item: GoldenItem, answer: AnswerResponse) -> float:
    if item.category is QuestionCategory.NO_ANSWER or not item.source_docs:
        return 1.0 if item.category is QuestionCategory.NO_ANSWER and not answer.answered else 0.0
    retrieved_sources = {rc.chunk.source for rc in answer.retrieved}
    hits = sum(1 for src in item.source_docs if src in retrieved_sources)
    return hits / len(item.source_docs)


def score_citation_accuracy(answer: AnswerResponse) -> Optional[float]:
    """Fraction of citations verified as supporting their claims.

    None when verification never ran or failed (verified flags are unknown) -
    an unmeasured citation is not an inaccurate one.
    """
    if not answer.answered:
        return 1.0 if not answer.citations else None
    if "verification_failed" in answer.metadata.get("errors", []):
        return None
    if not answer.citations:
        return 0.0  # answered with zero citations = uncited answer
    if any(c.verified is None for c in answer.citations):
        return None
    verified = sum(1 for c in answer.citations if c.verified)
    return verified / len(answer.citations)
