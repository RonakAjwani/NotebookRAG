"""Render eval reports to JSON + markdown."""

from __future__ import annotations

import json
import os
from typing import List

from app.evals.schemas import EvalReport


def write_reports(reports: List[EvalReport], out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "eval_report.json")
    md_path = os.path.join(out_dir, "eval_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in reports], f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(reports))

    return json_path, md_path


def _render_markdown(reports: List[EvalReport]) -> str:
    lines = ["# Evaluation Report", ""]

    lines.append("## Summary")
    lines.append("")
    lines.append("| Strategy | Mode | N | Correctness | Faithfulness | Retrieval hit | Citation acc | Answered | Errored |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r.strategy} | {r.mode} | {r.n} | {r.correctness:.3f} | {r.faithfulness:.3f} "
            f"| {r.retrieval_hit:.3f} | {r.citation_accuracy:.3f} | {r.answered_rate:.3f} "
            f"| {r.errored_cases} |"
        )
    lines.append("")
    lines.append(
        "_Aggregates are means over cases whose judge calls succeeded "
        "(`measured` in the JSON gives per-metric counts); errored cases are "
        "excluded, never scored as zero._"
    )
    lines.append("")

    if len(reports) > 1:
        best = max(reports, key=lambda r: (r.correctness + r.faithfulness + r.retrieval_hit))
        lines.append(f"**Best overall (correctness+faithfulness+retrieval): `{best.strategy}`**")
        lines.append("")

    for r in reports:
        lines.append(f"## {r.strategy} / {r.mode} - by category")
        lines.append("")
        lines.append("| Category | N | Correctness | Faithfulness | Retrieval hit | Citation acc |")
        lines.append("|---|---|---|---|---|---|")
        for cat, m in sorted(r.by_category.items()):
            lines.append(
                f"| {cat} | {m['n']} | {m['correctness']:.3f} | {m['faithfulness']:.3f} "
                f"| {m['retrieval_hit']:.3f} | {m['citation_accuracy']:.3f} |"
            )
        lines.append("")

    return "\n".join(lines)
