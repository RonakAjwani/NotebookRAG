"""Full evaluation driver: retrieval-mode comparison + chunking comparison.

Writes each EvalReport to ../results/ as it completes (durable against
rate-limit interruptions; re-running replays finished LLM calls from cache).

Produces:
  results/retrieval_modes/{dense,sparse,hybrid}.json   (recursive index)
  results/chunking/{fixed,recursive,semantic}.json      (hybrid mode)
  results/raw/*                                          (per-report JSON)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.evals.dataset import load_dataset          # noqa: E402
from app.evals.runner import EvalAborted, run_eval  # noqa: E402
from app.ingestion.pipeline import ingest_paths     # noqa: E402

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
CORPUS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "document corpus"))


def save(report, subdir: str, name: str) -> None:
    d = os.path.join(RESULTS, subdir)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"  saved {subdir}/{name}.json  "
          f"(correctness={report.correctness:.3f} faithfulness={report.faithfulness:.3f} "
          f"retrieval_hit={report.retrieval_hit:.3f} citation_acc={report.citation_accuracy:.3f} "
          f"errored={report.errored_cases}/{report.n})")


def main() -> int:
    os.makedirs(RESULTS, exist_ok=True)
    dataset = load_dataset(os.path.join(os.path.dirname(__file__), "..", "data", "golden.json"))
    n = sum(1 for i in dataset.items if i.approved)
    print(f"Golden set: {n} approved items. Results -> {RESULTS}\n")

    started = time.time()

    try:
        # --- Part 1 (required deliverable): chunking comparison, hybrid mode ---
        # Ordered first so the primary deliverable lands even if the tail throttles.
        print("=== Part 1: chunking-strategy comparison (hybrid mode) ===")
        recursive_hybrid = None
        for strategy in ["recursive", "fixed", "semantic"]:
            print(f"[strategy={strategy}] re-ingesting corpus ...")
            ingest_paths([CORPUS], strategy=strategy, reset=True)
            print(f"[strategy={strategy}] running {n} questions (hybrid) ...")
            rep = run_eval(dataset, mode="hybrid", strategy_label=f"{strategy}/hybrid")
            save(rep, "chunking", strategy)
            if strategy == "recursive":
                recursive_hybrid = rep

        # --- Part 2: retrieval-mode comparison on the recursive index ---
        # Re-ingest recursive (last loop left 'semantic'); reuse recursive/hybrid.
        print("\n=== Part 2: retrieval-mode comparison (recursive chunking) ===")
        ingest_paths([CORPUS], strategy="recursive", reset=True)
        if recursive_hybrid is not None:
            save(recursive_hybrid, "retrieval_modes", "hybrid")
        for mode in ["dense", "sparse"]:
            print(f"[mode={mode}] running {n} questions ...")
            rep = run_eval(dataset, mode=mode, strategy_label=f"recursive/{mode}")
            save(rep, "retrieval_modes", mode)
    except EvalAborted as exc:
        # Fail loud, preserve what completed. Cache makes a resume cheap.
        print(f"\n*** RUN ABORTED (integrity guard) ***\n{exc}\n"
              f"Completed configs are saved in {RESULTS}. Re-run the same command "
              f"after quota reset to resume from cache.")
        return 2

    print(f"\nDone in {(time.time()-started)/60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
