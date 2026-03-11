from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus import iter_corpus_documents
from src.reranking import CrossEncoderReranker
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.faiss_dense import FaissDenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.postprocess import (
    build_hotpot_gold_diagnostics,
    deduplicate_documents,
    pack_title_diverse_documents,
    select_title_representatives,
    unique_title_count,
)
from src.retrieval.sharded_dense import ShardedFaissDenseRetriever
from src.retrieval.title_prefilter import DenseShardedTitlePrefilterRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA retrieval recall on local indexes.")
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=Path("data/raw/flashrag/hotpotqa/dev/qa.jsonl"),
        help="Path to normalized HotpotQA qa.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/indexes/wiki_passages/manifest.json"),
        help="Index manifest path.",
    )
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid", "dense_sharded", "dense_sharded_title_prefilter"],
        default="bm25",
    )
    parser.add_argument("--hybrid-alpha", type=float, default=0.5)
    parser.add_argument("--hybrid-rrf-k", type=int, default=60)
    parser.add_argument("--hybrid-candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=100,
        help="Candidate depth fetched before optional dedup. Use 0 to match --top-k.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used when the retriever supports retrieve_many().",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=["off", "title", "doc_id"],
        default="off",
        help="Optional retrieval dedup mode for evaluation experiments.",
    )
    parser.add_argument(
        "--dedup-before-rerank",
        action="store_true",
        help="Apply dedup before reranking instead of after reranking.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Apply cross-encoder reranking before final top-k evaluation.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--reranker-retriever-rank-weight", type=float, default=0.4)
    parser.add_argument("--reranker-rank-fusion-k", type=int, default=60)
    parser.add_argument(
        "--title-first-rerank",
        action="store_true",
        help="Rerank unique title representatives first, then repack final chunks by title diversity.",
    )
    parser.add_argument("--title-pool-k", type=int, default=40)
    parser.add_argument("--max-chunks-per-title", type=int, default=2)
    parser.add_argument("--min-unique-titles", type=int, default=6)
    parser.add_argument("--max-queries", type=int, default=200, help="Use <=0 to evaluate all.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Where to save metrics and per-query details.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Optional corpus path used to compute title coverage. Defaults to manifest corpus_path when present.",
    )
    parser.add_argument(
        "--coverage-cache-path",
        type=Path,
        default=None,
        help="Optional path for the persisted title coverage cache. Defaults next to the manifest.",
    )
    parser.add_argument(
        "--refresh-coverage-cache",
        action="store_true",
        help="Rebuild the persisted coverage title cache even if it already exists.",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=16,
        help="Override nprobe when using dense_sharded IVF indexes.",
    )
    parser.add_argument(
        "--title-bm25-manifest",
        type=Path,
        default=None,
        help="Title-only BM25 manifest used by dense_sharded_title_prefilter.",
    )
    parser.add_argument("--title-prefilter-bm25-path", type=Path, default=None)
    parser.add_argument("--title-prefilter-docstore-path", type=Path, default=None)
    parser.add_argument("--title-prefilter-k", type=int, default=30)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override shard search worker count for dense_sharded. Use <=0 for manifest/default.",
    )
    return parser.parse_args()


def _norm_title(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _load_qa(path: Path, max_queries: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if max_queries > 0 and len(rows) >= max_queries:
                break
    return rows


def _chunked(items: list[dict], size: int) -> list[list[dict]]:
    if size <= 0:
        raise ValueError(f"batch size must be > 0, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_retriever(
    manifest: dict,
    mode: str,
    *,
    nprobe: int | None,
    num_workers: int | None,
    dense_manifest_path: Path | None,
    title_bm25_manifest_path: Path | None,
    title_prefilter_bm25_path: Path | None,
    title_prefilter_docstore_path: Path | None,
    title_prefilter_k: int,
):
    if mode == "bm25":
        return BM25Retriever(manifest["bm25_path"], manifest["docstore_path"])
    if mode == "dense":
        return FaissDenseRetriever(
            manifest["faiss_index_path"],
            manifest["docstore_path"],
            manifest["dense_config_path"],
            manifest.get("docstore_offsets_path"),
        )
    if mode == "dense_sharded":
        return ShardedFaissDenseRetriever(
            manifest_path=manifest["manifest_path"],
            nprobe=nprobe,
            num_workers=num_workers,
        )
    if mode == "dense_sharded_title_prefilter":
        if dense_manifest_path is None:
            raise ValueError("dense_sharded_title_prefilter requires the dense manifest path")
        return DenseShardedTitlePrefilterRetriever(
            manifest_path=dense_manifest_path,
            title_bm25_manifest_path=title_bm25_manifest_path,
            title_prefilter_bm25_path=title_prefilter_bm25_path,
            title_prefilter_docstore_path=title_prefilter_docstore_path,
            title_prefilter_k=title_prefilter_k,
            nprobe=nprobe,
            num_workers=num_workers,
        )
    raise ValueError(f"Unsupported retriever mode: {mode}")


def _build_hybrid_retriever(manifest: dict, alpha: float, rrf_k: int, candidate_k: int) -> HybridRetriever:
    bm25 = BM25Retriever(manifest["bm25_path"], manifest["docstore_path"])
    dense = FaissDenseRetriever(
        manifest["faiss_index_path"],
        manifest["docstore_path"],
        manifest["dense_config_path"],
        manifest.get("docstore_offsets_path"),
    )
    return HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        alpha=alpha,
        rrf_k=rrf_k,
        candidate_k=candidate_k,
    )


def _resolve_coverage_title_sources(manifest: dict, fallback_corpus_path: Path | None) -> list[Path]:
    title_sources: list[Path] = []
    if "shards" in manifest:
        title_sources.extend(Path(shard["docstore_path"]) for shard in manifest.get("shards", []))
    elif manifest.get("docstore_path"):
        title_sources.append(Path(manifest["docstore_path"]))
    elif fallback_corpus_path is not None:
        title_sources.append(fallback_corpus_path)
    return title_sources


def _default_coverage_cache_path(manifest_path: Path) -> Path:
    return manifest_path.with_name("titles_cache.json.gz")


def _read_cache_payload(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_cache_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _load_or_build_coverage_titles(
    title_sources: list[Path],
    cache_path: Path,
    refresh: bool,
) -> tuple[set[str], str]:
    expected_sources = [str(path.resolve()) for path in title_sources]
    if not refresh and cache_path.exists():
        payload = _read_cache_payload(cache_path)
        cached_sources = [str(item) for item in payload.get("title_sources", [])]
        if cached_sources == expected_sources:
            titles = {str(title) for title in payload.get("titles", []) if str(title)}
            return titles, ",".join(expected_sources)

    titles: set[str] = set()
    for path in title_sources:
        for doc in iter_corpus_documents(path):
            if doc.title:
                titles.add(_norm_title(doc.title))
    _write_cache_payload(
        cache_path,
        {
            "title_sources": expected_sources,
            "titles": sorted(titles),
        },
    )
    return titles, ",".join(expected_sources)


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if args.retriever in {"dense_sharded", "dense_sharded_title_prefilter"}:
        manifest["manifest_path"] = str(args.manifest)

    qa_rows = _load_qa(args.qa_path, args.max_queries)
    corpus_path = args.corpus_path or (Path(manifest["corpus_path"]) if manifest.get("corpus_path") else None)
    coverage_sources = _resolve_coverage_title_sources(manifest, corpus_path)
    coverage_cache_path = args.coverage_cache_path or _default_coverage_cache_path(args.manifest)
    corpus_titles, coverage_source = _load_or_build_coverage_titles(
        coverage_sources,
        coverage_cache_path,
        refresh=bool(args.refresh_coverage_cache),
    )

    num_workers = args.num_workers if args.num_workers > 0 else None
    nprobe = args.nprobe if args.nprobe > 0 else None
    if args.retriever == "hybrid":
        retriever = _build_hybrid_retriever(
            manifest,
            alpha=args.hybrid_alpha,
            rrf_k=args.hybrid_rrf_k,
            candidate_k=args.hybrid_candidate_k,
        )
    else:
        retriever = _build_retriever(
            manifest,
            args.retriever,
            nprobe=nprobe,
            num_workers=num_workers,
            dense_manifest_path=args.manifest,
            title_bm25_manifest_path=args.title_bm25_manifest,
            title_prefilter_bm25_path=args.title_prefilter_bm25_path,
            title_prefilter_docstore_path=args.title_prefilter_docstore_path,
            title_prefilter_k=int(args.title_prefilter_k),
        )

    reranker = None
    if args.use_reranker:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            batch_size=args.reranker_batch_size,
            retriever_rank_weight=float(args.reranker_retriever_rank_weight),
            rank_fusion_k=int(args.reranker_rank_fusion_k),
        )

    hits = 0
    all_gold_hits = 0
    covered_any = 0
    covered_all = 0
    total = 0
    per_query: list[dict] = []
    latencies_ms: list[float] = []
    unique_title_counts: list[int] = []
    duplicate_removed_counts: list[int] = []
    rerank_latencies_ms: list[float] = []
    title_pool_counts: list[int] = []
    failure_bucket_counts: dict[str, int] = {}
    retrieve_top_k = args.retrieve_top_k if args.retrieve_top_k > 0 else args.top_k

    for batch in tqdm(_chunked(qa_rows, args.batch_size), desc=f"Evaluating {args.retriever}"):
        questions = [str(row.get("question", "")) for row in batch]
        start = time.perf_counter()
        if hasattr(retriever, "retrieve_many"):
            raw_docs_batch = retriever.retrieve_many(questions, retrieve_top_k)
        else:
            raw_docs_batch = [retriever.retrieve(question, retrieve_top_k) for question in questions]
        batch_retrieval_ms = (time.perf_counter() - start) * 1000
        per_query_retrieval_ms = batch_retrieval_ms / len(batch) if batch else 0.0

        for row, raw_docs in zip(batch, raw_docs_batch):
            question = str(row.get("question", ""))
            gold_titles = [_norm_title(t) for t in row.get("gold_titles", [])]
            coverage_any = any(gt in corpus_titles for gt in gold_titles if gt) if corpus_titles else False
            coverage_all = (
                bool(gold_titles) and all(gt in corpus_titles for gt in gold_titles if gt)
                if corpus_titles
                else False
            )
            latencies_ms.append(per_query_retrieval_ms)

            if args.dedup_mode != "off" and args.dedup_before_rerank:
                candidate_docs = deduplicate_documents(raw_docs, args.dedup_mode)
            else:
                candidate_docs = list(raw_docs)

            if args.title_first_rerank:
                candidate_docs = select_title_representatives(candidate_docs, max_titles=args.title_pool_k)

            start_rerank = time.perf_counter()
            if reranker is not None:
                ranked_docs = reranker.rerank(question, candidate_docs, top_k=len(candidate_docs))
            else:
                ranked_docs = list(candidate_docs)
            rerank_ms = (time.perf_counter() - start_rerank) * 1000 if reranker is not None else 0.0
            rerank_latencies_ms.append(rerank_ms)

            if args.dedup_mode != "off" and not args.dedup_before_rerank:
                deduped_docs = deduplicate_documents(ranked_docs, args.dedup_mode)
            else:
                deduped_docs = list(ranked_docs)

            if args.title_first_rerank:
                docs = pack_title_diverse_documents(
                    ranked_title_docs=ranked_docs,
                    raw_candidate_docs=raw_docs,
                    top_k=args.top_k,
                    max_chunks_per_title=args.max_chunks_per_title,
                    min_unique_titles=args.min_unique_titles,
                )
            else:
                docs = deduped_docs[: args.top_k]

            retrieved_titles = [_norm_title(getattr(d, "title", "")) for d in docs]
            retrieved_title_set = set(retrieved_titles)
            hit = any(gt in retrieved_title_set for gt in gold_titles if gt)
            hit_all = bool(gold_titles) and all(gt in retrieved_title_set for gt in gold_titles if gt)
            hits += 1 if hit else 0
            all_gold_hits += 1 if hit_all else 0
            covered_any += 1 if coverage_any else 0
            covered_all += 1 if coverage_all else 0
            total += 1
            unique_titles = unique_title_count(docs)
            unique_title_counts.append(unique_titles)
            duplicate_removed = max(0, len(raw_docs) - len(deduped_docs))
            duplicate_removed_counts.append(duplicate_removed)
            title_pool_counts.append(len(candidate_docs))
            second_gold_missing = len(gold_titles) >= 2 and gold_titles[1] not in retrieved_title_set
            gold_diagnostics = build_hotpot_gold_diagnostics(
                gold_titles=row.get("gold_titles", []),
                raw_candidates=raw_docs,
                deduped_candidates=deduped_docs,
                final_docs=docs,
            )
            failure_bucket_counts[gold_diagnostics.retrieval_failure_bucket] = (
                failure_bucket_counts.get(gold_diagnostics.retrieval_failure_bucket, 0) + 1
            )

            per_query.append(
                {
                    "id": row.get("id"),
                    "question": question,
                    "gold_titles": row.get("gold_titles", []),
                    "raw_candidate_count": len(raw_docs),
                    "dedup_mode": args.dedup_mode,
                    "dedup_candidate_count": len(deduped_docs),
                    "duplicate_candidates_removed": duplicate_removed,
                    "title_pool_count": len(candidate_docs),
                    "retrieved_titles": [d.title for d in docs],
                    "unique_retrieved_titles": unique_titles,
                    "coverage_any_gold_title": coverage_any,
                    "coverage_all_gold_titles": coverage_all,
                    "hit_at_k": hit,
                    "hit_all_gold_titles_at_k": hit_all,
                    "second_gold_missing": second_gold_missing,
                    "latency_ms": per_query_retrieval_ms,
                    "rerank_latency_ms": rerank_ms,
                    "gold_title_ranks": gold_diagnostics.gold_title_ranks,
                    "gold_titles_in_raw_candidates": gold_diagnostics.gold_titles_in_raw_candidates,
                    "gold_titles_after_dedup": gold_diagnostics.gold_titles_after_dedup,
                    "gold_titles_in_final_top_k": gold_diagnostics.gold_titles_in_final_top_k,
                    "missing_gold_count": gold_diagnostics.missing_gold_count,
                    "first_gold_found": gold_diagnostics.first_gold_found,
                    "second_gold_found": gold_diagnostics.second_gold_found,
                    "retrieval_failure_bucket": gold_diagnostics.retrieval_failure_bucket,
                }
            )

    recall = hits / total if total else 0.0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    metrics = {
        "retriever": args.retriever,
        "top_k": args.top_k,
        "retrieve_top_k": retrieve_top_k,
        "batch_size": int(args.batch_size),
        "dedup_mode": args.dedup_mode,
        "dedup_before_rerank": bool(args.dedup_before_rerank),
        "use_reranker": bool(args.use_reranker),
        "title_first_rerank": bool(args.title_first_rerank),
        "title_pool_k": int(args.title_pool_k),
        "max_chunks_per_title": int(args.max_chunks_per_title),
        "min_unique_titles": int(args.min_unique_titles),
        "num_queries": total,
        "CoverageAny": (covered_any / total if total else 0.0),
        "CoverageAll": (covered_all / total if total else 0.0),
        "hit_queries": hits,
        "hit_all_gold_queries": all_gold_hits,
        "Recall@k_title": recall,
        "RecallAnyGoldTitle@k": recall,
        "RecallAllGold@k_title": (all_gold_hits / total if total else 0.0),
        "AvgRetrievalLatencyMs": avg_latency,
        "AvgRerankLatencyMs": (sum(rerank_latencies_ms) / len(rerank_latencies_ms) if rerank_latencies_ms else 0.0),
        "AvgUniqueTitles@k": (sum(unique_title_counts) / len(unique_title_counts) if unique_title_counts else 0.0),
        "AvgTitlePoolCount": (sum(title_pool_counts) / len(title_pool_counts) if title_pool_counts else 0.0),
        "AvgDuplicateCandidatesRemoved": (
            sum(duplicate_removed_counts) / len(duplicate_removed_counts) if duplicate_removed_counts else 0.0
        ),
        "FailureBucketCounts": failure_bucket_counts,
        "CoverageCachePath": str(coverage_cache_path),
        "CoverageSource": coverage_source,
    }
    if corpus_path is not None:
        metrics["CorpusPath"] = str(corpus_path)
    if args.retriever == "hybrid":
        metrics["hybrid_alpha"] = args.hybrid_alpha
        metrics["hybrid_rrf_k"] = args.hybrid_rrf_k
        metrics["hybrid_candidate_k"] = args.hybrid_candidate_k
    if args.use_reranker:
        metrics["RerankerModel"] = args.reranker_model
        metrics["RerankerRetrieverRankWeight"] = float(args.reranker_retriever_rank_weight)
        metrics["RerankerRankFusionK"] = int(args.reranker_rank_fusion_k)
    if hasattr(retriever, "retrieval_mode"):
        metrics["RetrievalMode"] = retriever.retrieval_mode
    if hasattr(retriever, "nprobe"):
        metrics["NProbe"] = retriever.nprobe
    if hasattr(retriever, "num_workers"):
        metrics["NumWorkers"] = retriever.num_workers
    if hasattr(retriever, "title_prefilter_k"):
        metrics["TitlePrefilterK"] = retriever.title_prefilter_k
    if hasattr(retriever, "dense_probe_top_k"):
        metrics["TitlePrefilterDenseProbeTopK"] = retriever.dense_probe_top_k
    if hasattr(retriever, "title_bm25_manifest_path"):
        metrics["TitleBm25ManifestPath"] = retriever.title_bm25_manifest_path
    if hasattr(retriever, "title_prefilter_bm25_path"):
        metrics["TitlePrefilterBm25Path"] = retriever.title_prefilter_bm25_path
    if hasattr(retriever, "title_prefilter_docstore_path"):
        metrics["TitlePrefilterDocstorePath"] = retriever.title_prefilter_docstore_path

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"hotpot_retrieval_{args.retriever}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (out_dir / "details.json").open("w", encoding="utf-8") as f:
        json.dump(per_query, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()
