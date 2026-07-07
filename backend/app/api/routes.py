"""/v1 API: ask, ingest, documents.

Thin HTTP layer over the pipeline. Answer generation runs in a threadpool
(`run_in_threadpool`) because the underlying LLM/embedding calls are synchronous
and would otherwise block the event loop.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.config import settings
from app.generation.generator import answer_question
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_paths
from app.models.schemas import AnswerResponse, DocumentInfo, IngestResponse, RetrievalMode
from app.retrieval.vector_store import get_store

router = APIRouter(prefix="/v1", tags=["rag"])


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    mode: RetrievalMode = RetrievalMode.HYBRID
    verify_citations: bool = True


@router.post("/ask", response_model=AnswerResponse)
async def ask(request: AskRequest) -> AnswerResponse:
    try:
        return await run_in_threadpool(
            answer_question,
            request.question,
            request.mode,
            verify_citations=request.verify_citations,
        )
    except Exception as exc:  # surface pipeline errors as 500 with a message
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: List[UploadFile] = File(...),
    strategy: Optional[str] = None,
    reset: bool = False,
) -> IngestResponse:
    tmp_dir = tempfile.mkdtemp(prefix="ingest_")
    saved: List[str] = []
    try:
        for f in files:
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")
            dest = os.path.join(tmp_dir, os.path.basename(f.filename or "upload"))
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest)
        return await run_in_threadpool(ingest_paths, saved, strategy, reset)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents() -> List[DocumentInfo]:
    store = get_store()
    store.ensure_collection()
    docs = await run_in_threadpool(store.list_documents)
    return [
        DocumentInfo(
            doc_id=d["doc_id"], source=d["source"], chunk_count=d["chunk_count"], strategies=d["strategies"]
        )
        for d in docs
    ]


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str) -> None:
    store = get_store()
    await run_in_threadpool(store.delete_document, doc_id)
