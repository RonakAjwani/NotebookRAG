"""Golden dataset + eval result shapes."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    LOOKUP = "lookup"          # single-fact retrieval
    MULTI_HOP = "multi_hop"    # needs info from >1 document
    NO_ANSWER = "no_answer"    # not answerable from the corpus
    AMBIGUOUS = "ambiguous"    # underspecified / multiple readings


class GoldenItem(BaseModel):
    id: str
    question: str
    answer: str = Field(..., description="Reference answer; empty for no_answer items")
    category: QuestionCategory
    source_docs: List[str] = Field(default_factory=list, description="Expected supporting docs")
    approved: bool = Field(False, description="Human-reviewed and accepted into the golden set")


class GoldenDataset(BaseModel):
    corpus: str = Field("", description="Description of the source corpus")
    items: List[GoldenItem] = Field(default_factory=list)


class CaseResult(BaseModel):
    id: str
    category: QuestionCategory
    question: str
    mode: str
    answered: bool
    # None = the judge for this metric failed; excluded from aggregates.
    correctness: Optional[float] = None
    faithfulness: Optional[float] = None
    retrieval_hit: float = 0.0     # objective (no LLM): expected docs in retrieved set?
    citation_accuracy: Optional[float] = None
    confidence: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    strategy: str
    mode: str
    n: int
    # Aggregates are means over cases where the metric was actually measured.
    correctness: float
    faithfulness: float
    retrieval_hit: float
    citation_accuracy: float
    answered_rate: float
    errored_cases: int = 0          # cases with any judge/pipeline error
    measured: dict = Field(default_factory=dict)  # metric -> how many cases counted
    by_category: dict = Field(default_factory=dict)
    cases: List[CaseResult] = Field(default_factory=list)
