"""Seed the index with a sample corpus so a fresh deployment is queryable.

    python -m scripts.seed                 # ingest ./sample_corpus
    python -m scripts.seed ./my_docs       # ingest a different folder

Safe to run repeatedly: chunk ids are deterministic and dedup skips repeats.
Replace sample_corpus/ with your own internal docs for a real deployment.
"""

from __future__ import annotations

import os
import sys

# Allow running as `python scripts/seed.py` too.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.pipeline import ingest_paths  # noqa: E402


def main() -> int:
    corpus = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "sample_corpus")
    corpus = os.path.abspath(corpus)
    if not os.path.isdir(corpus):
        print(f"Corpus directory not found: {corpus}")
        return 1
    print(f"Seeding index from {corpus} ...")
    result = ingest_paths([corpus])
    print(f"Indexed {result.chunks_indexed} chunks from {len(result.documents)} documents "
          f"({result.duplicates_skipped} duplicates skipped).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
