from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.postprocess import normalize_title
from src.retrieval.sharded_dense import ShardedFaissDenseRetriever
from src.types import Document


class DenseShardedTitlePrefilterRetriever:
    """Reorders deep dense candidates using a title-only BM25 lexical prior."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        title_bm25_manifest_path: str | Path | None = None,
        title_prefilter_bm25_path: str | Path | None = None,
        title_prefilter_docstore_path: str | Path | None = None,
        title_prefilter_k: int = 30,
        dense_probe_top_k: int = 300,
        nprobe: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        if title_prefilter_k <= 0:
            raise ValueError(f"title_prefilter_k must be > 0, got {title_prefilter_k}")
        if dense_probe_top_k <= 0:
            raise ValueError(f"dense_probe_top_k must be > 0, got {dense_probe_top_k}")

        manifest_file = Path(manifest_path)
        self._dense = ShardedFaissDenseRetriever(
            manifest_path=manifest_file,
            nprobe=nprobe,
            num_workers=num_workers,
        )

        title_manifest = None
        if title_bm25_manifest_path is not None:
            with Path(title_bm25_manifest_path).open("r", encoding="utf-8") as f:
                title_manifest = json.load(f)

        bm25_path = Path(title_prefilter_bm25_path) if title_prefilter_bm25_path else None
        docstore_path = Path(title_prefilter_docstore_path) if title_prefilter_docstore_path else None
        if title_manifest is not None:
            if bm25_path is None:
                bm25_path = Path(title_manifest["bm25_path"])
            if docstore_path is None:
                docstore_path = Path(title_manifest["docstore_path"])

        if bm25_path is None or docstore_path is None:
            raise ValueError(
                "title prefilter requires either title_bm25_manifest_path or both "
                "title_prefilter_bm25_path and title_prefilter_docstore_path"
            )

        self._title_retriever = BM25Retriever(bm25_path=bm25_path, docstore_path=docstore_path)
        self._title_prefilter_k = int(title_prefilter_k)
        self._dense_probe_top_k = int(dense_probe_top_k)
        self._retrieval_mode = "dense_sharded_title_prefilter"
        self._title_bm25_manifest_path = str(title_bm25_manifest_path) if title_bm25_manifest_path else ""
        self._title_prefilter_bm25_path = str(bm25_path)
        self._title_prefilter_docstore_path = str(docstore_path)

    @property
    def retrieval_mode(self) -> str:
        return self._retrieval_mode

    @property
    def nprobe(self) -> int | None:
        return self._dense.nprobe

    @property
    def num_workers(self) -> int:
        return self._dense.num_workers

    @property
    def title_prefilter_k(self) -> int:
        return self._title_prefilter_k

    @property
    def dense_probe_top_k(self) -> int:
        return self._dense_probe_top_k

    @property
    def title_bm25_manifest_path(self) -> str:
        return self._title_bm25_manifest_path

    @property
    def title_prefilter_bm25_path(self) -> str:
        return self._title_prefilter_bm25_path

    @property
    def title_prefilter_docstore_path(self) -> str:
        return self._title_prefilter_docstore_path

    def close(self) -> None:
        self._dense.close()

    def _rank_with_title_prefilter(self, query: str, dense_docs: list[Document], top_k: int) -> list[Document]:
        if top_k <= 0:
            return []
        if not dense_docs:
            return []

        title_docs = self._title_retriever.retrieve(query, self._title_prefilter_k)
        title_hits = {
            normalize_title(doc.title or doc.text or doc.doc_id)
            for doc in title_docs
            if (doc.title or doc.text or doc.doc_id).strip()
        }
        if not title_hits:
            return dense_docs[:top_k]

        prioritized: list[Document] = []
        fallback: list[Document] = []
        for doc in dense_docs:
            normalized = normalize_title(doc.title)
            if normalized and normalized in title_hits:
                prioritized.append(doc)
            else:
                fallback.append(doc)
        return (prioritized + fallback)[:top_k]

    def retrieve_many(self, queries: list[str], top_k: int) -> list[list[Document]]:
        if not queries:
            return []
        if top_k <= 0:
            return [[] for _ in queries]

        dense_k = max(int(top_k), self._dense_probe_top_k)
        dense_results = self._dense.retrieve_many(queries, top_k=dense_k)
        return [
            self._rank_with_title_prefilter(query, dense_docs, top_k=top_k)
            for query, dense_docs in zip(queries, dense_results)
        ]

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return self.retrieve_many([query], top_k=top_k)[0]
