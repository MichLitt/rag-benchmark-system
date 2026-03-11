from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.retrieval.docstore import load_docstore
from src.retrieval.tokenize import simple_tokenize
from src.types import Document


class BM25Retriever:
    def __init__(self, bm25_path: str | Path, docstore_path: str | Path) -> None:
        with Path(bm25_path).open("rb") as f:
            self._bm25 = pickle.load(f)
        self._docs = load_docstore(docstore_path)

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        query_tokens = simple_tokenize(query)
        scores = np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
        if scores.size == 0:
            return []
        k = min(top_k, int(scores.size))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self._docs[int(idx)] for idx in top_idx if int(idx) < len(self._docs)]
