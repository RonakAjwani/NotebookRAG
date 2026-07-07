# Hybrid RAG over Internal Docs

A production-grade Retrieval-Augmented Generation system that ingests internal
documentation, indexes it with **both dense vector and sparse BM25 search**,
retrieves the most relevant context with **Reciprocal Rank Fusion + LLM
reranking**, and generates **grounded answers with verified inline citations** -
plus a confidence score and a graceful "I don't know" when the evidence is thin.

## Highlights

- **Hybrid retrieval**: dense (fastembed `bge-small-en-v1.5`) + sparse (BM25) in a
  single Qdrant collection, fused with configurable-weight RRF, then reranked
  listwise by an LLM.
- **Grounded, verified answers**: every claim cites `[n]` sources; a judge LLM
  verifies each citation actually supports its claim (always on).
- **Confidence scoring**: retrieval quality, citation coverage, and completeness ->
  a composite score; low retrieval confidence triggers a structured "I don't know".
- **Three chunking strategies** (fixed / recursive / semantic) that can be
  compared head-to-head by the eval harness.
- **Eval-first**: a hand-curated golden Q&A set drives automated metrics
  (correctness, faithfulness, retrieval hit, citation accuracy) - no manual poking.
- **Dashboard**: ask questions, see clickable citations, ranked retrieved chunks,
  confidence breakdown, and a hybrid-vs-dense side-by-side toggle.

## Architecture

![System architecture](architecture.svg)

The backend module map lives in [`backend/README.md`](backend/README.md).

## Tech stack

- **Language**: Python 3.12 (the ML stack lags newer releases - pin 3.12).
- **API**: FastAPI + Uvicorn.
- **Vector store**: Qdrant (named dense + sparse vectors per point).
- **Embeddings**: fastembed - `BAAI/bge-small-en-v1.5` (dense) + `Qdrant/bm25` (sparse), local, no API key.
- **Chunking**: LangChain text splitters + a semantic splitter.
- **LLM inference**: Cerebras (default) or Groq - OpenAI-compatible, swappable via env.
- **Frontend**: React 18 + TypeScript + Vite + shadcn/ui + Tailwind.
- **Deployment**: Docker Compose (api + qdrant + frontend).

## Quick start (Docker)

```bash
# 1. Add your LLM keys
cp backend/.env.example backend/.env
#    edit backend/.env: set CEREBRAS_API_KEY and GROQ_API_KEY

# 2. Bring up qdrant + backend + frontend
docker compose up --build

# 3. Seed the sample corpus (in another shell)
docker compose exec backend python -m scripts.seed

# App:      http://localhost:8080
# API docs: http://localhost:8000/docs
```

## Quick start (local dev)

```bash
# --- Qdrant ---
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# --- Backend ---
cd backend
python -m venv venv && ./venv/Scripts/Activate.ps1   # (Windows) or: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                  # add your keys
python -m scripts.seed                                # seed sample corpus
uvicorn app.main:app --reload --port 8000

# --- Frontend ---
cd frontend
npm install
npm run dev                                           # http://localhost:8080
```

## Ingesting & evaluating

```bash
cd backend

# Ingest your own docs (md / txt / html / pdf)
python -m app.ingest ./path/to/docs --strategy recursive

# Evaluation (see backend/README.md for the golden-dataset workflow)
python -m app.evals run --mode hybrid
python -m app.evals compare --mode hybrid     # fixed vs recursive vs semantic
```

## Evaluation results

Full analysis in [`evaluation_results/REPORT.md`](evaluation_results/REPORT.md).

The eval harness (`python -m scripts.run_full_eval`) ran the full ablation
matrix, three chunking strategies and three retrieval modes, over a privately
curated 24-doc / 210-chunk corpus with a 36-question hand-authored golden set
(lookup / multi-hop / no-answer / ambiguous categories). The corpus, golden
set, and raw per-case artifacts stay private; the report above is the
published record. A follow-up run on a public corpus with externally written
questions is planned.

### Retrieval-mode comparison (recursive chunking, 36 questions)

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

### Answerable questions only (29 items; unanswerable traps scored separately)

| config | correctness | faithfulness | retrieval_hit | citation_acc |
|---|---|---|---|---|
| hybrid (recursive) | 0.900 | 0.990 | **1.000** | 0.672 |
| hybrid (fixed) | 0.921 | 0.952 | 0.931 | 0.586 |
| hybrid (semantic) | 0.803 | 0.993 | 0.931 | 0.776 |
| dense | 0.752 | 0.993 | 0.793 | 0.684 |
| sparse | 0.848 | 0.955 | 0.914 | 0.569 |

On the 7 unanswerable trap questions, hybrid abstained 7/7 (fixed 7/7,
semantic 6/7); dense and sparse answered every one, since the refusal gate
needs the reranker's scores and is inert in those modes.

Hybrid retrieval was the only configuration that handled the whole task:
sparse BM25 matched it on keyword lookups but fell behind on multi-hop
retrieval, dense-only missed exact-keyword questions, and only hybrid could
abstain. Weak spots, measured and reported rather than hidden: citation
attribution is the weakest metric (0.67 on answered questions, a conservative
lower bound), ambiguous questions get answered under one interpretation, and
the refusal gate fails open when the reranker errors. Numbers are only
meaningful at this corpus scale and have not yet been reproduced without the
response cache; see the report for the full honest decomposition.
