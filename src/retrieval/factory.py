from __future__ import annotations

from typing import Any

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import FaissDenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.sharded_dense import ShardedFaissDenseRetriever
from src.retrieval.title_prefilter import DenseShardedTitlePrefilterRetriever
from src.types import Document


def build_retriever(cfg: dict[str, Any], corpus: list[Document]):
    retrieval_cfg = cfg.get("retrieval", {})
    mode = str(retrieval_cfg.get("mode", "keyword")).lower()

    if mode == "keyword":
        return KeywordRetriever(corpus)

    if mode == "bm25":
        return BM25Retriever(
            bm25_path=retrieval_cfg["bm25_path"],
            docstore_path=retrieval_cfg["docstore_path"],
        )

    if mode == "dense":
        return FaissDenseRetriever(
            faiss_index_path=retrieval_cfg["faiss_index_path"],
            docstore_path=retrieval_cfg["docstore_path"],
            dense_config_path=retrieval_cfg["dense_config_path"],
            docstore_offsets_path=retrieval_cfg.get("docstore_offsets_path"),
        )

    if mode == "dense_sharded":
        return ShardedFaissDenseRetriever(
            manifest_path=retrieval_cfg["dense_shards_manifest_path"],
            nprobe=(
                int(retrieval_cfg["nprobe"])
                if retrieval_cfg.get("nprobe") not in (None, "")
                else None
            ),
            num_workers=(
                int(retrieval_cfg["num_workers"])
                if retrieval_cfg.get("num_workers") not in (None, "")
                else None
            ),
        )

    if mode == "dense_sharded_title_prefilter":
        return DenseShardedTitlePrefilterRetriever(
            manifest_path=retrieval_cfg["dense_shards_manifest_path"],
            title_bm25_manifest_path=retrieval_cfg.get("title_prefilter_manifest_path"),
            title_prefilter_bm25_path=retrieval_cfg.get("title_prefilter_bm25_path"),
            title_prefilter_docstore_path=retrieval_cfg.get("title_prefilter_docstore_path"),
            title_prefilter_k=int(retrieval_cfg.get("title_prefilter_k", 30)),
            dense_probe_top_k=int(retrieval_cfg.get("title_prefilter_dense_probe_top_k", 300)),
            nprobe=(
                int(retrieval_cfg["nprobe"])
                if retrieval_cfg.get("nprobe") not in (None, "")
                else None
            ),
            num_workers=(
                int(retrieval_cfg["num_workers"])
                if retrieval_cfg.get("num_workers") not in (None, "")
                else None
            ),
        )

    if mode == "hybrid":
        bm25 = BM25Retriever(
            bm25_path=retrieval_cfg["bm25_path"],
            docstore_path=retrieval_cfg["docstore_path"],
        )
        dense = FaissDenseRetriever(
            faiss_index_path=retrieval_cfg["faiss_index_path"],
            docstore_path=retrieval_cfg["docstore_path"],
            dense_config_path=retrieval_cfg["dense_config_path"],
            docstore_offsets_path=retrieval_cfg.get("docstore_offsets_path"),
        )
        return HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            alpha=float(retrieval_cfg.get("hybrid_alpha", 0.5)),
            rrf_k=int(retrieval_cfg.get("hybrid_rrf_k", 60)),
            candidate_k=int(retrieval_cfg.get("hybrid_candidate_k", 50)),
        )

    raise ValueError(f"Unsupported retrieval.mode: {mode}")
