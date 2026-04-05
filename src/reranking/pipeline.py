"""Two-stage reranking pipeline (B2).

Stage 1 — Cross-encoder coarse pass
    :class:`~src.reranking.cross_encoder.CrossEncoderReranker` reduces a large
    candidate pool (e.g. top-100) to a manageable set (e.g. top-20).

Stage 2 — Setwise LLM fine pass
    :class:`~src.reranking.setwise.SetwiseLLMReranker` refines the Stage-1
    output to the desired *top_k* using an LLM (e.g. Qwen2.5-3B-Instruct
    running locally).  The LLM call size is bounded by
    ``SetwiseLLMReranker.max_candidates`` (default 10) so context overrun is
    avoided for small models.

Typical configuration (from the Phase 2 plan)::

    Stage 1: top-100 → top-20   (cross-encoder coarse pass)
    Stage 2: top-20 → top-5     (setwise LLM fine pass, ≤10 per call)
"""
from __future__ import annotations

import logging

from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.setwise import SetwiseLLMReranker
from src.types import Document

logger = logging.getLogger(__name__)


class TwoStageReranker:
    """Cross-encoder coarse pass followed by setwise LLM fine pass.

    Args:
        cross_encoder: Stage-1 :class:`CrossEncoderReranker`.
        setwise: Stage-2 :class:`SetwiseLLMReranker`.
        coarse_top_k: How many candidates Stage-1 passes to Stage-2.
            Should be ≤ ``setwise.max_candidates`` × number-of-windows to
            avoid excessive LLM calls.  Default is 20 (plan §3.2 B2).
    """

    def __init__(
        self,
        cross_encoder: CrossEncoderReranker,
        setwise: SetwiseLLMReranker,
        coarse_top_k: int = 20,
    ) -> None:
        if coarse_top_k <= 0:
            raise ValueError(f"coarse_top_k must be > 0, got {coarse_top_k}")
        self._ce = cross_encoder
        self._sw = setwise
        self._coarse_k = coarse_top_k

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """Two-stage rerank returning top-k documents."""
        reranked, _ = self.rerank_with_scores(query, docs, top_k)
        return reranked

    def rerank_with_scores(
        self, query: str, docs: list[Document], top_k: int
    ) -> tuple[list[Document], list[float]]:
        """Two-stage rerank returning ``(documents, scores)``."""
        if not docs or top_k <= 0:
            return [], []

        # Stage 1: cross-encoder coarse pass
        stage1_k = min(self._coarse_k, len(docs))
        logger.debug(
            "TwoStageReranker stage-1: %d → %d candidates", len(docs), stage1_k
        )
        stage1_docs = self._ce.rerank(query, docs, top_k=stage1_k)

        # Stage 2: setwise LLM fine pass
        logger.debug(
            "TwoStageReranker stage-2: %d → %d candidates", len(stage1_docs), top_k
        )
        return self._sw.rerank_with_scores(query, stage1_docs, top_k=top_k)
