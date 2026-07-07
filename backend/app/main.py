"""FastAPI application entry point for the hybrid RAG service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as v1_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the Qdrant collection exists on startup. Best-effort: the API should
    # still boot (e.g. for /health) if Qdrant isn't reachable yet.
    try:
        from app.retrieval.vector_store import get_store

        get_store().ensure_collection()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] Qdrant not ready ({exc}). Collection will be created on first use.")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description="Hybrid-search RAG over internal documentation with grounded, cited answers.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)


@app.get("/")
async def root():
    return {"name": settings.app_name, "version": settings.api_version, "docs": "/docs", "status": "online"}


@app.get("/health")
async def health():
    return {"status": "healthy", "environment": settings.environment}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)
