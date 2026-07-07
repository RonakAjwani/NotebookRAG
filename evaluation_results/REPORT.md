# Hybrid RAG - Final Evaluation Report

**Run date:** 2026-07-07. **Duration:** 45.1 min. **Golden set:** 36 hand-authored questions (all `approved: true`). **Corpus:** 24 markdown notes, 210 chunks (recursive).

> **Dataset note:** the corpus and the golden Q&A set were privately curated from internal project notes and are not included in this repository, nor are the raw per-case result JSONs derived from them. This report is the published record of the run; a follow-up evaluation on a public corpus with externally written questions is planned.

This is the final clean run after the eval-harness hardening (silent-failure fixes, citation-measurement fixes, integrity guards). Every number below was **measured, never fabricated**: failed judge calls are recorded as `None` and excluded from aggregates (the `measured` counts in each JSON confirm all 36 cases were scored for every metric in every config), and an abort guard would have killed the run rather than emit garbage.

---

## 1. What was evaluated

The pipeline: **dense (bge-small) + sparse (BM25) retrieval -> weighted RRF fusion -> listwise LLM rerank -> grounded generation with inline `[n]` citations -> always-on claim-level citation verification -> composite confidence score -> "I don't know" refusal gate.**

Two ablation axes, 5 distinct configurations (`retrieval_modes/hybrid.json` is the same run as `chunking/recursive.json`, saved to both for convenience):

- **Retrieval mode** (recursive chunking): `hybrid` vs `dense`-only vs `sparse`-only. Dense/sparse skip the reranker - which matters (see section 4).
- **Chunking strategy** (hybrid mode): `recursive` vs `fixed` vs `semantic`, each on a freshly re-ingested index of the identical corpus.

**Model routing** (judge/generator separation to avoid self-preference bias): generation + rerank on `cerebras/gpt-oss-120b`; citation verification + eval judging on `cerebras/gemma-4-31b` (different model family -> genuine cross-model judging).

**Metrics.** `correctness` and `faithfulness` are LLM-as-judge against the hand-written reference (no_answer items are scored objectively: abstain = 1.0, answer = 0.0). `retrieval_hit` is programmatic - fraction of the expected source docs present in the retrieved set (for no_answer items it encodes "did the system abstain"). `citation_accuracy` is the fraction of an answer's citation markers that the independent verifier confirmed support their claims; an answer with zero citations scores 0.0; a correct abstention scores 1.0.

---

## 2. Headline results

### Retrieval-mode comparison (recursive index, 36 questions)

| mode | correctness | faithfulness | retrieval_hit | citation_acc | answered_rate | errored |
|---|---|---|---|---|---|---|
| **hybrid** | **0.919** | 0.992 | **1.000** | **0.736** | 0.806 | 1/36 |
| dense | 0.606 | 0.994 | 0.639 | 0.551 | 1.000 | 0/36 |
| sparse | 0.683 | 0.964 | 0.736 | 0.458 | 1.000 | 0/36 |

### Chunking-strategy comparison (hybrid mode, 36 questions)

| strategy | correctness | faithfulness | retrieval_hit | citation_acc | errored |
|---|---|---|---|---|---|
| **recursive** | 0.919 | 0.992 | **1.000** | 0.736 | 1/36 |
| fixed | **0.936** | 0.961 | 0.944 | 0.667 | 1/36 |
| semantic | 0.814 | **0.994** | 0.917 | **0.792** | 2/36 |

---

## 3. The honest decomposition

The aggregates above mix answerable questions with the 7 no_answer items, whose objective scoring (abstain -> 1.0 across the board) flatters any config that abstains well. Splitting them apart:

### Answerable questions only (29 items: 14 lookup, 10 multi_hop, 5 ambiguous)

| config | correctness | faithfulness | retrieval_hit | citation_acc |
|---|---|---|---|---|
| hybrid (recursive) | 0.900 | 0.990 | **1.000** | 0.672 |
| hybrid (fixed) | 0.921 | 0.952 | 0.931 | 0.586 |
| hybrid (semantic) | 0.803 | 0.993 | 0.931 | 0.776 |
| dense | 0.752 | 0.993 | 0.793 | 0.684 |
| sparse | 0.848 | 0.955 | 0.914 | 0.569 |

### no_answer questions (7 items): did it abstain?

| config | abstained |
|---|---|
| hybrid (recursive) | **7/7** |
| hybrid (fixed) | **7/7** |
| hybrid (semantic) | 6/7 |
| dense | 0/7 |
| sparse | 0/7 |

Two implications worth stating plainly:

1. **The headline citation accuracy of 0.736 includes 7 automatic 1.0s from correct abstentions.** On actually-answered questions it is **0.672** - matching the ~0.67 conservative lower bound established during the harness fixes. Report the 0.67 figure when talking about attribution quality.
2. **Hybrid's win is two separate wins.** On answerable questions it beats dense by +0.15 correctness and +0.21 retrieval hit (sparse is closer: +0.05 / +0.09). The rest of the aggregate gap comes from abstention: hybrid refuses all 7 unanswerable questions; dense and sparse answer every one of them (answered_rate 1.000 - they fabricate an answer for every trap question, though faithfulness ~0.99 shows the fabrications are at least grounded in whatever was retrieved).

---

## 4. Findings

### 4.1 Hybrid wins - but say precisely what won

Headline: retrieval hit 1.000 vs 0.639 (dense) / 0.736 (sparse); correctness 0.919 vs 0.606 / 0.683. Two caveats before quoting those gaps:

- **This is a pipeline comparison, not a pure retrieval ablation.** The mode switch changes two things at once: which index is queried AND whether the reranker + refusal gate exist (dense/sparse run without either). The no_answer items alone account for roughly half of the correctness gap (see section 3), and the ablation modes *cannot* abstain by construction (section 4.3).
- **On answerable questions only, the fair gaps are:** hit 1.000 vs 0.793 (dense) / 0.914 (sparse); correctness 0.900 vs 0.752 / 0.848. Plain BM25 is genuinely competitive on this keyword-rich corpus - hybrid's edge over sparse alone is modest (+0.05 correctness, +0.09 hit) and concentrated in multi_hop (sparse 0.75 hit / 0.81 correctness vs hybrid 1.00 / 0.93).

The defensible claim: **hybrid is the only configuration that does everything at once** - perfect on lookups (like sparse), strong on multi-hop (unlike either single index), and able to abstain (structurally unavailable to the ablation modes). No single index comes close on the full task; on answerable questions alone, sparse is nearer than the headline table suggests.

### 4.2 BM25 nails keyword lookups; dense embeddings don't

Per-category, sparse scores **1.000/1.000** (hit/correctness) on the 14 keyword-heavy lookup questions - identical to hybrid - while dense manages only 0.786/0.786. Dense misses the three lookups whose answers hinge on exact terminology (lk02, lk10, lk13). The reverse holds on multi_hop: dense 0.70 hit vs sparse 0.75, both well under hybrid's 1.00 - fusion is what covers both failure modes at once. This asymmetry is the empirical justification for hybrid search.

### 4.3 The refusal gate works - with three caveats stated plainly

Hybrid abstained on 7/7 unanswerable questions while answering all 29 answerable ones - zero false abstentions. But:

1. **Dense/sparse abstaining 0/7 is a mathematical certainty, not a measurement.** Without rerank scores, `retrieval_confidence` falls back to fused scores *normalized by their own max* - the top score becomes 1.0 and the blend (0.6*max + 0.4*mean) is always >= 0.6, which can never fall below the 0.35 gate threshold. The gate is a no-op in the ablation modes. Their 0/7 is an architectural property of the pipeline, not evidence that dense retrieval "can't tell" a trap question.
2. **One of the 7 abstentions cleared the gate by 0.01.** na01's retrieval confidence was 0.34 against the 0.35 threshold (composite 0.136 = 0.4 x 0.34) - and that same question flipped to (wrongly) answered in the semantic config. Perfect separation is real in this run but fragile at n=7; don't present it as a robust margin.
3. **The gate fails open on rerank failure.** The gate only runs when the rerank succeeded (`rerank_ok`); when the rerank judge fails, generation proceeds regardless - semantic/na01 is exactly this path. Failing closed (abstain when rerank fails on weak fused evidence) would be the safer design.

### 4.4 Chunking: recursive is the right default; semantic - the fanciest - loses

- **Recursive** is the only strategy with perfect retrieval (1.000) and it holds the best correctness/faithfulness balance (0.919/0.992).
- **Fixed** posts the top correctness (0.936) but the gap over recursive comes almost entirely from the 5-question ambiguous category (0.80 vs 0.56) - n=5 with a 0.3-granularity judge is noise territory. Meanwhile it pays real costs: retrieval misses (0.944), the worst lookup faithfulness (0.90), and its one rerank failure (lk10) cascaded into a fully wrong answer (correctness 0.0).
- **Semantic** underperforms where it should shine: multi_hop correctness 0.71 (vs 0.93 recursive / 0.97 fixed), 2/36 errored cases, and the only abstention failure. Its one bright spot - the top citation accuracy (0.792) - is only ~2 cases above recursive's 0.736 on a bimodal metric (section 5.1), i.e. within noise; don't lean on it. The most complex strategy measurably loses: a finding worth reporting precisely because it's counterintuitive.

**Decision: recursive stays the default.**

### 4.5 Confidence calibration: strong on groundedness, blind to interpretation errors

Mean composite confidence by bucket (hybrid/recursive):

| bucket | n | mean confidence |
|---|---|---|
| abstained (no_answer) | 7 | 0.02 |
| answered, all citations verified | 19 | 0.87 |
| answered, zero citations verified | 9 | 0.57 |
| answered, correctness >= 0.8 | 24 | 0.77 |
| answered, correctness <= 0.5 | 5 | 0.80 |

The score cleanly separates abstentions (~0) and tracks citation verification (0.87 vs 0.57 - the coverage dimension works). But it does **not** discriminate correctness among answered questions - confidently-cited wrong answers exist. All five low-correctness answered cases are ambiguous/underspecified questions where the system picked one valid interpretation, answered it well-groundedly (faithfulness 1.0), and got docked by the judge for not matching the reference's interpretation (e.g. am02 "What are the three strategies?" -> confidence 0.95, correctness 0.5). Confidence measures *"is this answer grounded and attributed"*, not *"did we resolve the question's ambiguity correctly"*. An intent-clarification step, not better retrieval, is the fix - retrieval hit was 1.000 on every ambiguous question.

There is also a second, sharper failure mode: **confidence inflates when the rerank fails.** The same self-normalizing fallback from section 4.3 runs inside the hybrid path on rerank failure, so the retrieval dimension reads ~1.0 regardless of actual evidence quality. Concretely: recursive/mh10 - rerank failed, half-wrong answer, confidence **0.979, the single highest confidence in the entire run**; fixed/lk10 - rerank failed, wrong document retrieved, fully wrong answer, confidence 0.636 with faithfulness 1.0 (perfectly grounded in the wrong text). `rerank_ok = False` should cap or flag confidence, not inflate it.

### 4.6 Error accounting (nothing hidden)

4 distinct errored cases across the 3 hybrid configs (108 reranked cases -> **3.7% rerank JSON-failure rate** on gpt-oss-120b), zero judge or verification failures, zero excluded metrics (`measured` = 36/36 everywhere). Rerank failures degrade gracefully to fused order but are not free: fixed/lk10 produced a wrong answer, recursive/mh10 a half-credit one, semantic/na01 broke the refusal gate. Dense/sparse had zero errors (no reranker to fail).

### 4.7 Reproducibility - what "matched the prior run" actually means

Correctness/faithfulness/hit figures matched the prior 94-min clean run because the prompt-hash cache **replayed the same generations and judgments - identical by construction, not an independent replication**. (Temp-0 calls make a fresh rerun likely to land close, but that has not been demonstrated; a true replication would require clearing the cache.) The citation-accuracy figures are the genuinely new measurement in this run, taken after the three citation-extraction fixes (bracket-variant normalization, claim-level granularity against the union of cited chunks, conservative-verifier acknowledgment): 0.47 -> 0.672 answered-only, an honest lower bound.

---

## 5. Limitations (reported, not hidden)

1. **Citation accuracy 0.672 on answered questions** is the weak spot - and the per-case scores are essentially **bimodal** (hybrid: 19 cases at 1.0, 9 at 0.0, 1 at 0.5), so config-level differences of ±0.1 on this metric are 2-3 cases, i.e. noise. Three known contributors to the 0.0s: the generator sometimes answers correctly with *zero* citation markers (scored 0.0; all 9 zero-citation-score cases in hybrid still had correctness >= 0.8 except the two ambiguous ones); occasional genuine attribution imprecision; and a deliberately conservative verifier (gemma-4-31b produces occasional false negatives on clearly-supported citations - kept because an independent under-counting judge beats a self-graded inflating one). Faithfulness at 0.99 shows the answers *are* grounded; the gap is per-claim attribution, not hallucination.
2. **The 9 zero-score citation cases cannot be audited from the saved artifacts.** The result JSONs persist scores but not answer text or citation lists (the `results/raw/` output mentioned in `run_full_eval.py`'s docstring was never implemented), so "emitted no `[n]` markers" vs "all markers failed verification" - two very different stories - cannot be distinguished after the fact. For a project whose headline feature is citation verification, this is a real auditability gap.
3. **Scale and difficulty make the perfect numbers cheaper than they look.** 24 documents / 210 chunks; `retrieval_hit` is *document-level* presence in the top-5 (not passage-level); questions were hand-authored by reading the docs and often reuse the source's exact phrasing - ideal BM25 bait. Retrieval hit 1.000 is genuinely measured but is weak evidence about production-scale retrieval. Also: 8 of the 24 docs are this project's own design notes (a partially self-referential corpus), and everything is markdown - the PDF/HTML loaders were never exercised.
4. **No held-out set, single-author risk.** Thresholds, fusion weights, and prompts were developed alongside the same 36 questions used for the final numbers, and the golden set was hand-authored by the same assistant that built the system. Nothing was knowingly tuned to the set, but in-sample optimism cannot be ruled out without fresh unseen questions.
5. **Faithfulness (~0.99) is near-ceiling and non-discriminative.** Each config gets 7 automatic 1.0s from correct abstentions, and the metric measures grounding in *whatever was retrieved*: fixed/lk10 scored faithfulness 1.0 on a fully wrong answer faithfully grounded in the wrong document. Don't showcase this number as "no hallucination."
6. **Ambiguous-question correctness 0.56** (hybrid) - the system answers one interpretation instead of surfacing the ambiguity. Retrieval is not the bottleneck (hit 1.000).
7. **Confidence has two blind spots** (section 4.5): it doesn't flag interpretation errors (wrong-but-grounded answers score high), and it *inflates* on rerank failure (mh10: 0.979 on a half-wrong answer - the run's highest confidence).
8. **The refusal gate fails open on rerank failure, is a no-op in dense/sparse by construction, and cleared na01 by 0.01** (section 4.3).
9. **~3.7% rerank JSON failures** even on the reliable model; degradation is graceful but not free (one fully wrong answer, one half-credit, one broken abstention).
10. **Small n everywhere:** 36 questions, n=5 in the smallest category - sub-category deltas below ~0.2 are judge noise (e.g. fixed's ambiguous-category "win" over recursive).

### Cheap follow-ups that would strengthen the story

- Persist per-case answer text + citation verdicts in the eval output and re-emit (the cache makes a re-run nearly free) - makes the citation 0.0s auditable.
- Cap or flag confidence when `rerank_ok=False`; consider failing the gate closed on weak fused evidence.
- Author ~10 fresh questions *after* freezing the system (someone other than the system's author, ideally) as a held-out sanity check.
- Clear the LLM cache and re-run once to demonstrate true run-to-run stability rather than replay-by-construction.

---

## 6. Verdict

The numbers are real, honestly produced, and they support a more modest claim than the headline table suggests - state that claim and it holds up to scrutiny. On a small, keyword-rich, partially self-referential corpus, the full hybrid pipeline is the only configuration that handles the entire task: sparse-level perfection on lookups, the only strong multi-hop retrieval, and the only working abstention - while plain BM25 comes surprisingly close on answerable questions (0.848 vs 0.900 correctness), which is itself an honest and useful finding. The chunking comparison yields a defensible, slightly counterintuitive engineering decision (recursive over semantic, with the fanciest strategy losing).

What earns credibility is everything around the numbers: fail-loud judges, `None`-excluded metrics, an abort guard, measured-count reporting, cross-model judging - and the willingness to publish the weak spots as findings: citation attribution 0.67 (bimodal, partly unauditable from artifacts), ambiguity handling 0.56, a fail-open gate with a 0.01-margin abstention, confidence that inflates on rerank failure, no held-out set. For a portfolio, present it as *"a measurement-first RAG build whose eval was hardened until it stopped lying, including about itself"* - that is both true and more impressive than the inflated version.

---

*The raw per-config JSONs (per-case scores, per-category rollups, `measured` counts) are kept privately alongside the corpus and golden set. The harness itself is in this repo: `python -m scripts.run_full_eval` reproduces the full matrix against any corpus + golden set placed at `backend/data/golden.json` (Qdrant on :6333, Cerebras + Groq keys in `backend/.env`).*
