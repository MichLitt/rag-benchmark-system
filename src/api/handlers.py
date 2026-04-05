"""Request handlers and the ScoredDocument adapter for the retrieval API.

Design principle (from Phase 2 plan §A2):
  ``ScoredDocument`` is **only** constructed here in the API layer.
  The existing retrieval pipeline (pipeline.py, cross_encoder.py, hybrid.py)
  continues to operate on plain ``List[Document]`` — no internal interfaces
  are changed.
"""
from __future__ import annotations

import time

from fastapi import HTTPException

from src.api.index_registry import IndexRegistry
from src.api.models import (
    HealthResponse,
    ResultMetadata,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveResultItem,
)
from src.logging_utils import get_logger
from src.types import Document, ScoredDocument

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Registry injection point
# ---------------------------------------------------------------------------

_registry: IndexRegistry | None = None


def get_registry() -> IndexRegistry:
    """Return the active IndexRegistry, creating a default one if needed."""
    global _registry
    if _registry is None:
        _registry = IndexRegistry()
    return _registry


def set_registry(reg: IndexRegistry | None) -> None:
    """Replace the active registry (used in tests to inject a tmp-dir registry)."""
    global _registry
    _registry = reg


# ---------------------------------------------------------------------------
# Adapter: wrap plain Documents with rank-based scores
# ---------------------------------------------------------------------------

def _rank_scores(n: int) -> list[float]:
    """Return rank-based proxy scores 1/(rank+1) when the retriever hides scores."""
    return [1.0 / (rank + 1) for rank in range(n)]


def _wrap_scores(
    docs: list[Document],
    scores: list[float],
    stage: str,
) -> list[ScoredDocument]:
    """Zip docs + scores into ScoredDocument objects (API-layer adapter)."""
    return [
        ScoredDocument(document=doc, score=score, retrieval_stage=stage)
        for doc, score in zip(docs, scores)
    ]


# ---------------------------------------------------------------------------
# Handler: POST /v1/retrieve
# ---------------------------------------------------------------------------

def handle_retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Synchronous retrieve handler — called via ``run_in_threadpool`` from server.py."""
    reg = get_registry()

    # Validate index_id (check both already-loaded and still-on-disk)
    available = set(reg.available_index_ids()) | set(reg.loaded_index_ids())
    if req.index_id not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Index {req.index_id!r} not found. "
                   f"Available: {sorted(available) or '(none)'}",
        )

    t0 = time.perf_counter()

    try:
        retriever = reg.get_retriever(req.index_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    docs: list[Document] = retriever.retrieve(req.query, req.top_k)

    # Reranker path (MVP: only if use_reranker requested AND a reranker is available)
    if req.use_reranker and docs:
        try:
            from src.reranking.cross_encoder import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            docs, raw_scores = reranker.rerank_with_scores(req.query, docs, req.top_k)
            stage = "rerank"
            scores = raw_scores
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranker unavailable (%s); falling back to base retrieval.", exc)
            scores = _rank_scores(len(docs))
            stage = reg.index_type(req.index_id)
    else:
        scores = _rank_scores(len(docs))
        stage = reg.index_type(req.index_id)

    scored = _wrap_scores(docs, scores, stage)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    results = [
        RetrieveResultItem(
            doc_id=sd.document.doc_id,
            text=sd.document.text,
            score=sd.score,
            metadata=ResultMetadata(
                title=sd.document.title or None,
                source=sd.document.source,
                page_start=sd.document.page_start,
                page_end=sd.document.page_end,
                section=sd.document.section,
            ),
        )
        for sd in scored
    ]

    return RetrieveResponse(
        results=results,
        latency_ms=latency_ms,
        retrieval_profile=req.retrieval_profile,
        index_id=req.index_id,
    )


# ---------------------------------------------------------------------------
# Handler: GET /v1/health
# ---------------------------------------------------------------------------

def handle_health() -> HealthResponse:
    reg = get_registry()
    return HealthResponse(
        status="ok",
        indexes_loaded=reg.loaded_index_ids(),
    )
