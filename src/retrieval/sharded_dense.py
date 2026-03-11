from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.docstore import LazyDocstore, load_docstore
from src.types import Document


def _normalize_index_type(value: str | None) -> str:
    raw = str(value or "flat").strip().lower()
    if raw.startswith("dense_sharded_"):
        raw = raw.removeprefix("dense_sharded_")
    if raw not in {"flat", "ivf_flat"}:
        return "flat"
    return raw


def _set_index_runtime_params(index: faiss.Index, index_type: str, nprobe: int | None) -> None:
    if index_type != "ivf_flat" or nprobe is None:
        return
    if nprobe <= 0:
        raise ValueError(f"nprobe must be > 0, got {nprobe}")
    try:
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", int(nprobe))
    except RuntimeError:
        if hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)


@dataclass
class _DenseShard:
    index: faiss.Index
    docstore: LazyDocstore | list[Document]
    index_type: str
    dimension: int
    nlist: int | None = None

    def get_doc(self, index: int) -> Document | None:
        if isinstance(self.docstore, LazyDocstore):
            if index < 0 or index >= len(self.docstore):
                return None
            return self.docstore.get(index)
        if index < 0 or index >= len(self.docstore):
            return None
        return self.docstore[index]


class ShardedFaissDenseRetriever:
    """Dense retriever that merges results from multiple FAISS shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        nprobe: int | None = None,
        num_workers: int | None = None,
    ) -> None:
        manifest_file = Path(manifest_path)
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self._manifest_path = manifest_file
        self._embedding_model = str(manifest["embedding_model"])
        self._model = SentenceTransformer(self._embedding_model)
        self._retrieval_mode = _normalize_index_type(manifest.get("index_type"))
        default_nprobe = manifest.get("nprobe")
        if default_nprobe in (None, "") and self._retrieval_mode == "ivf_flat":
            default_nprobe = 16
        self._nprobe = int(nprobe) if nprobe is not None else (int(default_nprobe) if default_nprobe else None)
        self._shards: list[_DenseShard] = []

        for shard_cfg in manifest.get("shards", []):
            index = faiss.read_index(str(shard_cfg["faiss_index_path"]))
            shard_index_type = _normalize_index_type(
                shard_cfg.get("index_type") or manifest.get("index_type")
            )
            shard_nprobe = shard_cfg.get("nprobe")
            effective_nprobe = self._nprobe
            if effective_nprobe is None and shard_nprobe not in (None, ""):
                effective_nprobe = int(shard_nprobe)
            _set_index_runtime_params(index, shard_index_type, effective_nprobe)

            offsets_path = shard_cfg.get("docstore_offsets_path")
            if offsets_path and Path(offsets_path).exists():
                docstore: LazyDocstore | list[Document] = LazyDocstore(
                    shard_cfg["docstore_path"],
                    offsets_path,
                )
            else:
                docstore = load_docstore(shard_cfg["docstore_path"])
            self._shards.append(
                _DenseShard(
                    index=index,
                    docstore=docstore,
                    index_type=shard_index_type,
                    dimension=int(shard_cfg.get("dimension") or manifest.get("dimension") or 0),
                    nlist=int(shard_cfg["nlist"]) if shard_cfg.get("nlist") not in (None, "") else None,
                )
            )

        inferred_workers = min(len(self._shards), cpu_count() or 1) if self._shards else 1
        manifest_workers = manifest.get("num_workers")
        self._num_workers = (
            int(num_workers)
            if num_workers is not None
            else int(manifest_workers) if manifest_workers not in (None, "") else inferred_workers
        )
        if self._num_workers <= 0:
            raise ValueError(f"num_workers must be > 0, got {self._num_workers}")

    @property
    def retrieval_mode(self) -> str:
        return self._retrieval_mode

    @property
    def nprobe(self) -> int | None:
        return self._nprobe

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def close(self) -> None:
        for shard in self._shards:
            if isinstance(shard.docstore, LazyDocstore):
                shard.docstore.close()

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        emb = self._model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32, copy=False)

    def _search_shard(
        self,
        shard_idx: int,
        shard: _DenseShard,
        embeddings: np.ndarray,
        top_k: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        scores, indices = shard.index.search(embeddings, top_k)
        return shard_idx, scores, indices

    def retrieve_many(self, queries: list[str], top_k: int) -> list[list[Document]]:
        if not queries:
            return []
        if top_k <= 0:
            return [[] for _ in queries]

        embeddings = self._encode_queries(queries)
        shard_results: list[tuple[int, np.ndarray, np.ndarray]] = []
        if len(self._shards) <= 1 or self._num_workers <= 1:
            for shard_idx, shard in enumerate(self._shards):
                shard_results.append(self._search_shard(shard_idx, shard, embeddings, top_k))
        else:
            max_workers = min(self._num_workers, len(self._shards))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._search_shard, shard_idx, shard, embeddings, top_k)
                    for shard_idx, shard in enumerate(self._shards)
                ]
                shard_results = [future.result() for future in futures]

        merged_per_query: list[list[tuple[float, int, int]]] = [[] for _ in queries]
        for shard_idx, scores, indices in shard_results:
            for query_idx in range(len(queries)):
                for score, doc_idx in zip(scores[query_idx].tolist(), indices[query_idx].tolist()):
                    if doc_idx < 0:
                        continue
                    merged_per_query[query_idx].append((float(score), shard_idx, int(doc_idx)))

        query_docs: list[list[Document]] = []
        for merged in merged_per_query:
            merged.sort(key=lambda item: item[0], reverse=True)
            docs: list[Document] = []
            for _, shard_idx, doc_idx in merged[:top_k]:
                doc = self._shards[shard_idx].get_doc(doc_idx)
                if doc is None:
                    continue
                docs.append(doc)
            query_docs.append(docs)
        return query_docs

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return self.retrieve_many([query], top_k=top_k)[0]
