# Backend - Hybrid RAG

FastAPI service implementing the ingestion -> hybrid retrieval -> grounded
generation -> evaluation pipeline.

## Module map

```
app/
├── config.py              # pydantic-settings; every tunable knob (env-driven)
├── main.py                # FastAPI app + lifespan (ensures Qdrant collection)
├── ingest.py              # CLI: python -m app.ingest <paths> [--strategy] [--reset]
├── models/schemas.py      # Chunk, RetrievedChunk, Citation, AnswerResponse, ...
├── llm/
│   ├── client.py          # provider-agnostic LLM client: retry/backoff + cache
│   └── cache.py           # prompt-hash response cache (resume-and-skip)
├── ingestion/
│   ├── loaders.py         # md / txt / html / pdf -> Blocks (section + page meta)
│   ├── chunkers.py        # fixed | recursive | semantic strategies
│   ├── dedup.py           # sha256 exact + cosine>0.95 near-dup skip
│   └── pipeline.py        # load -> persist raw+processed -> chunk -> embed -> dedup -> index
├── retrieval/
│   ├── embeddings.py      # fastembed dense (bge) + sparse (bm25)
│   ├── vector_store.py    # Qdrant wrapper: one collection, named dense+sparse vectors
│   ├── fusion.py          # weighted Reciprocal Rank Fusion
│   ├── reranker.py        # listwise LLM reranker (RankGPT-style, one call/query)
│   └── retriever.py       # orchestration + dense|sparse|hybrid mode switch
├── generation/
│   ├── prompts.py         # numbered-context, cite-[n], refuse-if-unsupported
│   ├── verification.py    # per-answer batched citation verification (always on)
│   ├── confidence.py      # retrieval / coverage / completeness -> composite
│   └── generator.py       # full answer pipeline + "I don't know" gate
├── evals/                 # golden dataset, LLM-as-judge metrics, runner, reports
└── api/routes.py          # /v1/ask, /v1/ingest, /v1/documents
```

## Key design points

- **Providers are swappable.** Cerebras / Groq / OpenAI are all OpenAI-compatible;
  `LLM_PROVIDER` selects the generator and `JUDGE_PROVIDER` selects a *different*
  model for reranking / verification / evals (avoids self-preference bias).
- **The LLM client is resilient by construction.** Exponential backoff on 429s so a
  rate-limit graze never crashes a run, plus a prompt-hash cache so re-runs replay
  unchanged calls for free. This is what makes the eval-first workflow affordable.
- **Dense + sparse stay in sync automatically** because both vectors live on the
  same Qdrant point - one upsert writes both, one delete removes both.
- **Citation verification is always on**, batched one call per answer (every
  claim-citation pair judged together), because the confidence score's
  citation-coverage dimension depends on it.

## Evaluation workflow (curated golden set)

The golden Q&A set is **hand-authored** from a real sample of the target docs
(reviewed for quality), then consumed by the runner. Shape lives in
`app/evals/schemas.py` (`GoldenItem`: question, answer, category, one of
lookup / multi_hop / no_answer / ambiguous, source_docs, approved).

```bash
# Place the curated golden set at backend/data/golden.json (items with
# "approved": true are the ones that count).
python -m app.evals run --mode hybrid       # correctness / faithfulness / retrieval / citation
python -m app.evals compare --mode hybrid   # re-ingest per strategy, compare fixed/recursive/semantic
```

`app/evals/dataset.py` also provides an optional `generate` command that *drafts*
questions with an LLM as a starting point - but the intended path is a curated,
human-authored set (drafts still require review + `approved: true`).

## Configuration

All settings are in `app/config.py` and overridable via `backend/.env`
(see `.env.example`). Notable knobs: `CHUNK_STRATEGY`, `RRF_DENSE_WEIGHT` /
`RRF_SPARSE_WEIGHT`, `RERANK_TOP_K`, `RETRIEVAL_CONFIDENCE_THRESHOLD`.

## Requirements

Python 3.12. A running Qdrant (via `docker compose` or a standalone container).
LLM keys are only needed for `/v1/ask` and the eval harness - ingestion and
retrieval embeddings run locally without any keys.
