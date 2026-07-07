"""Golden Q&A dataset generation from the processed corpus.

LLM-drafted, human-approved. Known failure mode of generated questions: they
reuse the source's exact vocabulary and become trivially retrievable. Mitigations
applied here:
  - a paraphrase instruction (ask in different words than the passage),
  - category-specific generation (multi_hop deliberately samples two *different*
    documents; no_answer asks for plausible-but-absent questions),
  - every item lands with approved=False; a human must review before it counts.
"""

from __future__ import annotations

import glob
import json
import os
import random
import uuid
from typing import Dict, List

from app.config import settings
from app.evals.schemas import GoldenDataset, GoldenItem, QuestionCategory
from app.llm.client import get_generator

_SYSTEM = "You write evaluation questions for a documentation QA system. Output strict JSON."


def _load_processed() -> Dict[str, str]:
    """doc source name -> concatenated text, from data/processed/*.json."""
    docs: Dict[str, str] = {}
    for path in glob.glob(os.path.join(settings.processed_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = "\n\n".join(b["text"] for b in data.get("blocks", []))
        docs[data.get("source", os.path.basename(path))] = text
    return docs


def _gen_json(prompt: str) -> dict:
    from app.llm.client import LLMError

    try:
        data = get_generator().complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return data if isinstance(data, dict) else {}
    except LLMError:
        return {}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _gen_lookup(source: str, text: str) -> GoldenItem | None:
    prompt = f"""From the passage below, write ONE factual lookup question and its answer.
Phrase the question in DIFFERENT words than the passage (avoid copying its exact terms).

Passage from "{source}":
{text[:2500]}

Return JSON: {{"question": "...", "answer": "..."}}"""
    d = _gen_json(prompt)
    if not d.get("question") or not d.get("answer"):
        return None
    return GoldenItem(
        id=_new_id(), question=d["question"], answer=d["answer"],
        category=QuestionCategory.LOOKUP, source_docs=[source],
    )


def _gen_multi_hop(s1: str, t1: str, s2: str, t2: str) -> GoldenItem | None:
    prompt = f"""Write ONE question whose answer REQUIRES combining information from BOTH
passages below (from two different documents). Provide the combined answer.
Phrase the question in different words than the passages.

Passage A (from "{s1}"):
{t1[:1500]}

Passage B (from "{s2}"):
{t2[:1500]}

Return JSON: {{"question": "...", "answer": "..."}}"""
    d = _gen_json(prompt)
    if not d.get("question") or not d.get("answer"):
        return None
    return GoldenItem(
        id=_new_id(), question=d["question"], answer=d["answer"],
        category=QuestionCategory.MULTI_HOP, source_docs=[s1, s2],
    )


def _gen_no_answer(source: str, text: str) -> GoldenItem | None:
    prompt = f"""Based on the TOPIC of the passage below, write ONE plausible question that a
user might ask but that is NOT answered anywhere in this passage. The answer must
NOT be derivable from it.

Passage from "{source}":
{text[:2000]}

Return JSON: {{"question": "..."}}"""
    d = _gen_json(prompt)
    if not d.get("question"):
        return None
    return GoldenItem(
        id=_new_id(), question=d["question"], answer="",
        category=QuestionCategory.NO_ANSWER, source_docs=[],
    )


def _gen_ambiguous(source: str, text: str) -> GoldenItem | None:
    prompt = f"""Write ONE underspecified / ambiguous question loosely related to the passage
(it could have multiple valid interpretations). Give the most reasonable answer.

Passage from "{source}":
{text[:2000]}

Return JSON: {{"question": "...", "answer": "..."}}"""
    d = _gen_json(prompt)
    if not d.get("question"):
        return None
    return GoldenItem(
        id=_new_id(), question=d["question"], answer=d.get("answer", ""),
        category=QuestionCategory.AMBIGUOUS, source_docs=[source],
    )


def generate_dataset(
    n_lookup: int = 12,
    n_multi_hop: int = 8,
    n_no_answer: int = 6,
    n_ambiguous: int = 4,
    seed: int = 13,
) -> GoldenDataset:
    docs = _load_processed()
    if not docs:
        raise RuntimeError(
            f"No processed documents found in {settings.processed_dir}. "
            "Run `python -m app.ingest <corpus>` first."
        )
    rng = random.Random(seed)
    names = list(docs.keys())
    items: List[GoldenItem] = []

    for _ in range(n_lookup):
        s = rng.choice(names)
        if (item := _gen_lookup(s, docs[s])):
            items.append(item)

    if len(names) >= 2:
        for _ in range(n_multi_hop):
            s1, s2 = rng.sample(names, 2)
            if (item := _gen_multi_hop(s1, docs[s1], s2, docs[s2])):
                items.append(item)

    for _ in range(n_no_answer):
        s = rng.choice(names)
        if (item := _gen_no_answer(s, docs[s])):
            items.append(item)

    for _ in range(n_ambiguous):
        s = rng.choice(names)
        if (item := _gen_ambiguous(s, docs[s])):
            items.append(item)

    return GoldenDataset(corpus=f"{len(names)} documents from {settings.processed_dir}", items=items)


def default_dataset_path() -> str:
    return os.path.join(os.path.dirname(settings.processed_dir.rstrip("/\\")), "golden.json")


def save_dataset(dataset: GoldenDataset, path: str | None = None) -> str:
    path = path or default_dataset_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, ensure_ascii=False, indent=2)
    return path


def load_dataset(path: str | None = None) -> GoldenDataset:
    path = path or default_dataset_path()
    with open(path, "r", encoding="utf-8") as f:
        return GoldenDataset(**json.load(f))
