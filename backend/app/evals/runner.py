"""Eval runner.

Processes the golden set SEQUENTIALLY, one question at a time through the exact
answer pipeline a live user hits (retrieve -> gate -> generate -> verify -> score).
This is deliberate: batching questions would measure a different system. The
prompt-hash cache in the LLM client makes re-runs cheap; retry/backoff keeps a run
alive through rate-limit grazes.

Integrity: judge failures are recorded per case and EXCLUDED from aggregates -
never scored as zero. If too many cases error, the run ABORTS loudly instead of
producing a report that looks like data. (The first eval run scored four whole
configs 0.0 after the judge provider's daily quota died; this guard exists so
that can never silently happen again.)
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import List, Optional

from app.evals import metrics
from app.evals.schemas import CaseResult, EvalReport, GoldenDataset
from app.generation.generator import answer_question
from app.models.schemas import RetrievalMode

# Abort when more than this fraction of processed cases carry errors.
ERROR_ABORT_FRACTION = 0.25
# Never abort before this many cases (avoid tripping on one flaky call).
ERROR_ABORT_MIN_CASES = 8


class EvalAborted(RuntimeError):
    pass


def _measured_mean(values: List[Optional[float]]) -> tuple[float, int]:
    real = [v for v in values if v is not None]
    return (round(mean(real), 4) if real else 0.0, len(real))


def run_eval(
    dataset: GoldenDataset,
    mode: RetrievalMode | str = RetrievalMode.HYBRID,
    strategy_label: str = "current",
    only_approved: bool = True,
) -> EvalReport:
    mode = RetrievalMode(mode)
    items = [it for it in dataset.items if it.approved or not only_approved]

    cases: List[CaseResult] = []
    for item in items:  # sequential, on purpose
        answer = answer_question(item.question, mode)
        errors = list(answer.metadata.get("errors", []))

        correctness, faithfulness = metrics.score_answer(item, answer)
        if correctness is None and faithfulness is None and answer.answered:
            errors.append("eval_judge_failed")
        citation_accuracy = metrics.score_citation_accuracy(answer)

        cases.append(
            CaseResult(
                id=item.id,
                category=item.category,
                question=item.question,
                mode=mode.value,
                answered=answer.answered,
                correctness=correctness,
                faithfulness=faithfulness,
                retrieval_hit=metrics.score_retrieval_hit(item, answer),
                citation_accuracy=citation_accuracy,
                confidence=answer.confidence.composite if answer.confidence else None,
                errors=errors,
            )
        )

        # Fail loud, early: a dying judge provider must stop the run, not
        # quietly zero out the rest of the suite.
        errored = sum(1 for c in cases if c.errors)
        if len(cases) >= ERROR_ABORT_MIN_CASES and errored / len(cases) > ERROR_ABORT_FRACTION:
            raise EvalAborted(
                f"{errored}/{len(cases)} cases have judge/pipeline errors "
                f"(> {ERROR_ABORT_FRACTION:.0%}). Aborting '{strategy_label}/{mode.value}' - "
                "likely provider quota exhaustion. Fix the provider (or wait for "
                "quota reset) and re-run; completed calls are cached."
            )

    correctness, n_corr = _measured_mean([c.correctness for c in cases])
    faithfulness, n_faith = _measured_mean([c.faithfulness for c in cases])
    citation_acc, n_cit = _measured_mean([c.citation_accuracy for c in cases])
    retrieval_hit, _ = _measured_mean([c.retrieval_hit for c in cases])

    by_cat: dict = {}
    grouped: dict = defaultdict(list)
    for c in cases:
        grouped[c.category.value].append(c)
    for cat, group in grouped.items():
        cat_corr, _ = _measured_mean([g.correctness for g in group])
        cat_faith, _ = _measured_mean([g.faithfulness for g in group])
        cat_hit, _ = _measured_mean([g.retrieval_hit for g in group])
        cat_cit, _ = _measured_mean([g.citation_accuracy for g in group])
        by_cat[cat] = {
            "n": len(group),
            "correctness": cat_corr,
            "faithfulness": cat_faith,
            "retrieval_hit": cat_hit,
            "citation_accuracy": cat_cit,
            "errored": sum(1 for g in group if g.errors),
        }

    return EvalReport(
        strategy=strategy_label,
        mode=mode.value,
        n=len(cases),
        correctness=correctness,
        faithfulness=faithfulness,
        retrieval_hit=retrieval_hit,
        citation_accuracy=citation_acc,
        answered_rate=round(mean([1.0 if c.answered else 0.0 for c in cases]), 4) if cases else 0.0,
        errored_cases=sum(1 for c in cases if c.errors),
        measured={
            "correctness": n_corr,
            "faithfulness": n_faith,
            "citation_accuracy": n_cit,
        },
        by_category=by_cat,
        cases=cases,
    )


def run_chunking_comparison(
    dataset: GoldenDataset,
    mode: RetrievalMode | str = RetrievalMode.HYBRID,
    strategies: List[str] | None = None,
) -> List[EvalReport]:
    """Re-ingest the raw corpus under each chunking strategy and eval each.

    Requires the raw corpus in settings.raw_dir (kept there at ingest time).
    """
    from app.config import settings
    from app.ingestion.pipeline import ingest_paths

    strategies = strategies or ["fixed", "recursive", "semantic"]
    reports: List[EvalReport] = []
    for strategy in strategies:
        ingest_paths([settings.raw_dir], strategy=strategy, reset=True)
        reports.append(run_eval(dataset, mode=mode, strategy_label=strategy))
    return reports
