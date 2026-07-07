"""CLI for the eval harness.

    # 1. Draft a golden dataset from the ingested corpus (then review it by hand)
    python -m app.evals generate

    # 2. Run the suite over the approved golden set
    python -m app.evals run --mode hybrid

    # 3. Compare all three chunking strategies (re-ingests the raw corpus each time)
    python -m app.evals compare --mode hybrid
"""

from __future__ import annotations

import argparse
import os
import sys

from app.config import settings
from app.evals.dataset import (
    default_dataset_path,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from app.evals.report import write_reports
from app.evals.runner import run_chunking_comparison, run_eval


def _reports_dir() -> str:
    base = os.path.dirname(settings.processed_dir.rstrip("/\\")) or "."
    return os.path.join(base, "eval_reports")


def cmd_generate(args) -> int:
    print("Drafting golden dataset from processed corpus ...")
    dataset = generate_dataset()
    path = save_dataset(dataset, args.out)
    approved = sum(1 for it in dataset.items if it.approved)
    print(f"Wrote {len(dataset.items)} draft items ({approved} approved) to {path}")
    print("Review the file and set \"approved\": true on the items you accept, then run:")
    print("  python -m app.evals run")
    return 0


def cmd_run(args) -> int:
    dataset = load_dataset(args.dataset)
    approved = [it for it in dataset.items if it.approved]
    if not approved:
        print("No approved items in the golden set. Set \"approved\": true on reviewed items first.")
        return 1
    print(f"Running eval over {len(approved)} approved items (mode={args.mode}) ...")
    report = run_eval(dataset, mode=args.mode)
    json_path, md_path = write_reports([report], _reports_dir())
    print(f"Correctness={report.correctness:.3f} Faithfulness={report.faithfulness:.3f} "
          f"RetrievalHit={report.retrieval_hit:.3f} CitationAcc={report.citation_accuracy:.3f}")
    print(f"Reports: {md_path}")
    return 0


def cmd_compare(args) -> int:
    dataset = load_dataset(args.dataset)
    if not any(it.approved for it in dataset.items):
        print("No approved items in the golden set.")
        return 1
    print(f"Comparing chunking strategies (mode={args.mode}) - re-ingests raw corpus each time ...")
    reports = run_chunking_comparison(dataset, mode=args.mode)
    json_path, md_path = write_reports(reports, _reports_dir())
    for r in reports:
        print(f"  {r.strategy}: correctness={r.correctness:.3f} faithfulness={r.faithfulness:.3f} "
              f"retrieval_hit={r.retrieval_hit:.3f}")
    print(f"Reports: {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.evals", description="RAG evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Draft a golden dataset from the corpus")
    g.add_argument("--out", default=default_dataset_path())
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="Run the eval suite")
    r.add_argument("--dataset", default=default_dataset_path())
    r.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="Compare chunking strategies")
    c.add_argument("--dataset", default=default_dataset_path())
    c.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
