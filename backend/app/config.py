"""Application configuration via pydantic-settings.

Every knob the pipeline exposes lives here so behaviour can be tuned from the
environment / .env without touching code. Grouped by pipeline stage.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App / server -----------------------------------------------------
    app_name: str = "Hybrid RAG API"
    api_version: str = "v1"
    environment: str = "development"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:8080,http://localhost:5173"

    # ---- Storage paths ----------------------------------------------------
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    cache_dir: str = "./data/cache"

    # ---- Qdrant -----------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "internal_docs"

    # ---- Embeddings (fastembed) ------------------------------------------
    # Dense model + its dimension must agree. bge-small-en-v1.5 => 384.
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dimension: int = 384
    sparse_model: str = "Qdrant/bm25"

    # ---- LLM inference (OpenAI-compatible providers) ----------------------
    # Default provider used for generation.
    llm_provider: Literal["cerebras", "groq", "openai", "openrouter"] = "cerebras"

    cerebras_api_key: str | None = None
    cerebras_base_url: str = "https://api.cerebras.ai/v1"
    cerebras_model: str = "gpt-oss-120b"

    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # Fallback provider for judge roles not routed explicitly below. Kept
    # separate from the generator so a model never grades its own output.
    judge_provider: Literal["cerebras", "groq", "openai", "openrouter"] = "groq"

    # Per-role routing, "provider" or "provider/model" (empty = judge_provider).
    # Free-tier quotas are per provider AND per model (each Cerebras model has
    # its own ~1M tokens/day pool), so spreading roles across models multiplies
    # the daily budget. Only gpt-oss-120b and gemma-4-31b reliably emit the
    # structured JSON these roles need; zai-glm-4.7 failed ~90% of rerank calls.
    # Rerank shares the generator's model because ranking passages is not
    # judging generated text; the roles that do judge generated text run on
    # gemma-4-31b, a different family from the generator. gemma is a slightly
    # conservative verifier (occasional false negatives), so citation accuracy
    # reads as a lower bound. Groq's 70B is more accurate but its 100K/day cap
    # aborts full runs; spot-audits only.
    rerank_llm: str = "cerebras/gpt-oss-120b"
    verify_llm: str = "cerebras/gemma-4-31b"
    completeness_llm: str = "cerebras/gemma-4-31b"
    eval_judge_llm: str = "cerebras/gemma-4-31b"

    # Proactive pacing: minimum spacing between calls per provider (requests/min).
    # Prevents 429 backoff storms instead of only reacting to them.
    cerebras_rpm: int = 5
    groq_rpm: int = 25
    openai_rpm: int = 60
    openrouter_rpm: int = 15

    # ---- LLM client behaviour --------------------------------------------
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_max_retries: int = 6
    llm_backoff_base: float = 2.0  # seconds; exponential
    llm_backoff_max: float = 60.0
    llm_cache_enabled: bool = True

    # ---- Chunking ---------------------------------------------------------
    # Default strategy used at ingest time when none is specified.
    chunk_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 800  # characters
    chunk_overlap: int = 150
    semantic_threshold: float = 0.75  # cosine breakpoint for semantic chunker

    # ---- Dedup ------------------------------------------------------------
    dedup_cosine_threshold: float = 0.95

    # ---- Retrieval --------------------------------------------------------
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60  # RRF smoothing constant
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rerank_candidates: int = 20  # fused list size sent to the reranker
    rerank_top_k: int = 5  # kept after rerank -> LLM context

    # ---- Generation / confidence -----------------------------------------
    # Below this mean rerank relevance, answer with the "I don't know" path.
    retrieval_confidence_threshold: float = 0.35

    @field_validator("rrf_dense_weight", "rrf_sparse_weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("RRF weights must be between 0 and 1")
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
