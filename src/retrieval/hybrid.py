from __future__ import annotations

from typing import TYPE_CHECKING

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import FaissDenseRetriever
from src.types import Document

if TYPE_CHECKING:
    from src.retrieval.splade import SPLADERetriever


# ---------------------------------------------------------------------------
# Standalone RRF helper (shared by HybridRetriever and DenseSpladeHybridRetriever)
# ---------------------------------------------------------------------------

def rrf_fusion(
    results_a: list[Document],
    results_b: list[Document],
    *,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    rrf_k: int = 60,
    top_k: int,
) -> list[Document]:
    """Reciprocal Rank Fusion of two ranked document lists.

    Each document's RRF score = ``weight * 1 / (rrf_k + rank)``, summed across
    lists.  Documents appearing in only one list still receive their RRF score
    from that list.

    Args:
        results_a: First ranked list (most relevant first).
        results_b: Second ranked list (most relevant first).
        weight_a: Weight applied to *results_a* RRF scores.
        weight_b: Weight applied to *results_b* RRF scores.
        rrf_k: RRF constant (default 60, from the original RRF paper).
        top_k: Maximum number of results to return.

    Returns:
        Merged and re-ranked list of up to *top_k* documents.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Document] = {}

    def _add(docs: list[Document], weight: float) -> None:
        for rank, doc in enumerate(docs, start=1):
            if not doc.doc_id:
                continue
            by_id[doc.doc_id] = doc
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + weight / (rrf_k + rank)

    _add(results_a, weight_a)
    _add(results_b, weight_b)
    ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [by_id[d] for d in ranked_ids[:top_k]]


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: FaissDenseRetriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
        if candidate_k <= 0:
            raise ValueError(f"candidate_k must be > 0, got {candidate_k}")

        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def _add_rrf_scores(
        self,
        docs: list[Document],
        weight: float,
        scores: dict[str, float],
        by_id: dict[str, Document],
    ) -> None:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.doc_id
            if not doc_id:
                continue
            by_id[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (weight / (self._rrf_k + rank))

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        if top_k <= 0:
            return []

        candidate_k = max(top_k, self._candidate_k)
        bm25_docs = self._bm25.retrieve(query, candidate_k)
        dense_docs = self._dense.retrieve(query, candidate_k)

        scores: dict[str, float] = {}
        by_id: dict[str, Document] = {}

        self._add_rrf_scores(bm25_docs, 1.0 - self._alpha, scores, by_id)
        self._add_rrf_scores(dense_docs, self._alpha, scores, by_id)

        ranked_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        return [by_id[doc_id] for doc_id in ranked_ids[:top_k]]


# ---------------------------------------------------------------------------
# Dense + SPLADE hybrid retriever (RRF fusion)
# ---------------------------------------------------------------------------

class DenseSpladeHybridRetriever:
    """Hybrid retriever that fuses dense (FAISS) and SPLADE sparse results via RRF.

    Args:
        dense_retriever: A :class:`FaissDenseRetriever` instance.
        splade_retriever: A :class:`~src.retrieval.splade.SPLADERetriever` instance.
        alpha: Weight for dense results in RRF (0 = SPLADE only, 1 = dense only).
        rrf_k: RRF constant (default 60).
        candidate_k: Candidates fetched from each retriever before fusion.
    """

    def __init__(
        self,
        dense_retriever: FaissDenseRetriever,
        splade_retriever: SPLADERetriever,
        alpha: float = 0.5,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
        if candidate_k <= 0:
            raise ValueError(f"candidate_k must be > 0, got {candidate_k}")
        self._dense = dense_retriever
        self._splade = splade_retriever
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        if top_k <= 0:
            return []
        ck = max(top_k, self._candidate_k)
        dense_docs: list[Document] = self._dense.retrieve(query, ck)
        splade_docs: list[Document] = self._splade.retrieve(query, ck)
        return rrf_fusion(
            dense_docs,
            splade_docs,
            weight_a=self._alpha,
            weight_b=1.0 - self._alpha,
            rrf_k=self._rrf_k,
            top_k=top_k,
        )
