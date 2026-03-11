from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from src.types import Document


def _argsort_desc_stable(values: np.ndarray) -> np.ndarray:
    return np.argsort(-values, kind="stable")


def rerank_documents_from_scores(
    docs: list[Document],
    scores: np.ndarray,
    top_k: int,
    retriever_rank_weight: float = 0.0,
    rank_fusion_k: int = 60,
) -> list[Document]:
    if top_k <= 0 or not docs:
        return []
    if not 0.0 <= retriever_rank_weight <= 1.0:
        raise ValueError(
            f"retriever_rank_weight must be in [0, 1], got {retriever_rank_weight}"
        )
    if rank_fusion_k <= 0:
        raise ValueError(f"rank_fusion_k must be > 0, got {rank_fusion_k}")

    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.shape[0] != len(docs):
        raise ValueError(
            f"scores length must match docs length, got {score_array.shape[0]} and {len(docs)}"
        )

    k = min(top_k, len(docs))
    rerank_order = _argsort_desc_stable(score_array)
    if retriever_rank_weight == 0.0:
        return [docs[int(i)] for i in rerank_order[:k]]

    rerank_ranks = np.empty(len(docs), dtype=np.int32)
    rerank_ranks[rerank_order] = np.arange(len(docs), dtype=np.int32)
    retrieval_ranks = np.arange(len(docs), dtype=np.int32)

    rerank_rrf = 1.0 / (rank_fusion_k + rerank_ranks + 1)
    retrieval_rrf = 1.0 / (rank_fusion_k + retrieval_ranks + 1)
    fused_scores = ((1.0 - retriever_rank_weight) * rerank_rrf) + (
        retriever_rank_weight * retrieval_rrf
    )
    fused_order = _argsort_desc_stable(fused_scores)
    return [docs[int(i)] for i in fused_order[:k]]


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        retriever_rank_weight: float = 0.0,
        rank_fusion_k: int = 60,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not 0.0 <= retriever_rank_weight <= 1.0:
            raise ValueError(
                f"retriever_rank_weight must be in [0, 1], got {retriever_rank_weight}"
            )
        if rank_fusion_k <= 0:
            raise ValueError(f"rank_fusion_k must be > 0, got {rank_fusion_k}")
        self._model = CrossEncoder(model_name)
        self._batch_size = batch_size
        self._retriever_rank_weight = retriever_rank_weight
        self._rank_fusion_k = rank_fusion_k

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if top_k <= 0 or not docs:
            return []

        pairs = [(query, f"{doc.title}\n{doc.text}".strip()) for doc in docs]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return rerank_documents_from_scores(
            docs=docs,
            scores=np.asarray(scores, dtype=np.float32),
            top_k=top_k,
            retriever_rank_weight=self._retriever_rank_weight,
            rank_fusion_k=self._rank_fusion_k,
        )
