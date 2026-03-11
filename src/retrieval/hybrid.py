from __future__ import annotations

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import FaissDenseRetriever
from src.types import Document


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
