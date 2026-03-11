from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.docstore import LazyDocstore, load_docstore
from src.types import Document


class FaissDenseRetriever:
    def __init__(
        self,
        faiss_index_path: str | Path,
        docstore_path: str | Path,
        dense_config_path: str | Path,
        docstore_offsets_path: str | Path | None = None,
    ) -> None:
        self._index = faiss.read_index(str(faiss_index_path))
        self._docstore = None
        if docstore_offsets_path is not None and Path(docstore_offsets_path).exists():
            self._docstore = LazyDocstore(docstore_path, docstore_offsets_path)
            self._docs = None
        else:
            self._docs = load_docstore(docstore_path)
        with Path(dense_config_path).open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_name = str(cfg["embedding_model"])
        self._model = SentenceTransformer(model_name)

    def close(self) -> None:
        if self._docstore is not None:
            self._docstore.close()

    def _get_doc(self, index: int) -> Document | None:
        if self._docstore is not None:
            if index < 0 or index >= len(self._docstore):
                return None
            return self._docstore.get(index)
        if index < 0 or self._docs is None or index >= len(self._docs):
            return None
        return self._docs[index]

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.astype(np.float32, copy=False)
        _, indices = self._index.search(emb, top_k)
        hit_docs: list[Document] = []
        for idx in indices[0].tolist():
            doc = self._get_doc(idx)
            if doc is None:
                continue
            hit_docs.append(doc)
        return hit_docs
