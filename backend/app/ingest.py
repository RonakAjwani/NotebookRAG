"""CLI: ingest a corpus into the vector store.

    python -m app.ingest ./corpus
    python -m app.ingest ./corpus --strategy semantic --reset
"""

from __future__ import annotations

import argparse
import sys

from app.ingestion.pipeline import ingest_paths
from app.models.schemas import ChunkStrategy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the hybrid RAG index.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkStrategy],
        default=None,
        help="Chunking strategy (defaults to CHUNK_STRATEGY from settings)",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Drop and recreate the collection first"
    )
    args = parser.parse_args(argv)

    print(f"Ingesting {args.paths} (strategy={args.strategy or 'default'}, reset={args.reset}) ...")
    result = ingest_paths(args.paths, strategy=args.strategy, reset=args.reset)

    for doc in result.documents:
        print(f"  {doc.source}: {doc.chunk_count} chunks [{', '.join(doc.strategies) or '-'}]")
    print(f"Indexed {result.chunks_indexed} chunks, skipped {result.duplicates_skipped} duplicates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
