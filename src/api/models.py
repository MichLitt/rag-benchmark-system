"""Pydantic request / response models for the retrieval API."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /v1/retrieve
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Natural-language retrieval query.")
    index_id: str = Field(..., description="Identifier of the pre-built index to search.")
    top_k: int = Field(5, ge=1, le=100, description="Maximum number of results to return.")
    retrieval_profile: str = Field(
        "auto",
        description=(
            "Named retrieval profile (e.g. 'bm25_v1', 'dense_v1'). "
            "Echoed back in the response for traceability. "
            "'auto' lets the registry pick the best available retriever."
        ),
    )
    use_reranker: bool = Field(
        False,
        description="Apply cross-encoder reranking on top of the base retrieval.",
    )


class ResultMetadata(BaseModel):
    title: str | None = None
    source: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


class RetrieveResultItem(BaseModel):
    doc_id: str
    text: str
    score: float = Field(..., description="Retrieval or rerank score (higher is better).")
    metadata: ResultMetadata


class RetrieveResponse(BaseModel):
    results: list[RetrieveResultItem]
    latency_ms: float = Field(..., description="End-to-end handler latency in milliseconds.")
    retrieval_profile: str = Field(..., description="Echo of the requested retrieval_profile.")
    index_id: str = Field(..., description="Echo of the requested index_id.")


# ---------------------------------------------------------------------------
# /v1/health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    indexes_loaded: list[str] = Field(
        default_factory=list,
        description="index_ids whose retrievers are currently in memory.",
    )
