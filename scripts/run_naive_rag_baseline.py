from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config
from src.evalops.adapter import build_eval_run_report
from src.evalops.client import EvalOpsClient
from src.logging_utils import configure_logging
from src.flashrag_loader import load_flashrag_qa
from src.generation import build_generator
from src.io_utils import append_run_result_jsonl, ensure_dir, save_json, save_run_results
from src.pipeline import run_naive_rag
from src.query import build_query_expander, resolve_query_expansion_mode
from src.reranking import CrossEncoderReranker
from src.retrieval.factory import build_retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.postprocess import normalize_title
from src.types import RunExampleResult


DATASET_PATHS = {
    "hotpotqa": Path("data/raw/flashrag/hotpotqa/dev/qa.jsonl"),
    "nq": Path("data/raw/flashrag/nq/test/qa.jsonl"),
    "triviaqa": Path("data/raw/flashrag/triviaqa/test/qa.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Naive RAG baseline on FlashRAG datasets.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "keyword", "hybrid", "dense_sharded", "dense_sharded_title_prefilter"],
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--generator-mode",
        choices=["extractive", "llm", "openai_compatible"],
        default=None,
        help="Generation backend. Defaults to config value.",
    )
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument("--generator-base-url", type=str, default=None)
    parser.add_argument("--generator-base-url-env", type=str, default=None)
    parser.add_argument("--generator-api-key-env", type=str, default=None)
    parser.add_argument("--generator-temperature", type=float, default=None)
    parser.add_argument("--generator-max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--generator-max-completion-tokens",
        type=int,
        default=None,
        help="Completion budget for reasoning-capable models such as MiniMax-M2.5.",
    )
    parser.add_argument(
        "--generator-reasoning-split",
        choices=["auto", "true", "false"],
        default=None,
        help="Whether to request separate reasoning output on compatible providers.",
    )
    parser.add_argument("--generator-timeout-sec", type=int, default=None)
    parser.add_argument("--generator-input-price-per-1m", type=float, default=None)
    parser.add_argument("--generator-output-price-per-1m", type=float, default=None)
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Apply cross-encoder reranker on retrieved candidates before generation.",
    )
    parser.add_argument(
        "--retrieve-top-k",
        type=int,
        default=None,
        help="Candidate depth fetched before optional dedup/reranking.",
    )
    parser.add_argument(
        "--dedup-mode",
        choices=["off", "title", "doc_id"],
        default=None,
        help="Optional retrieval dedup mode applied as an experiment.",
    )
    parser.add_argument(
        "--dedup-before-rerank",
        action="store_true",
        help="Apply dedup to retrieved candidates before reranking.",
    )
    parser.add_argument(
        "--title-first-rerank",
        action="store_true",
        help="Rerank unique title representatives first, then repack final chunks by title diversity.",
    )
    parser.add_argument(
        "--title-pool-k",
        type=int,
        default=None,
        help="Number of unique title representatives kept before reranking.",
    )
    parser.add_argument(
        "--max-chunks-per-title",
        type=int,
        default=None,
        help="Maximum number of chunks packed from each title in the final context.",
    )
    parser.add_argument(
        "--min-unique-titles",
        type=int,
        default=None,
        help="Target minimum number of unique titles preserved in final retrieved docs.",
    )
    parser.add_argument("--title-prefilter-manifest", type=Path, default=None)
    parser.add_argument("--title-prefilter-bm25-path", type=Path, default=None)
    parser.add_argument("--title-prefilter-docstore-path", type=Path, default=None)
    parser.add_argument("--title-prefilter-k", type=int, default=None)
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model name for reranking.",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument(
        "--reranker-retriever-rank-weight",
        type=float,
        default=0.0,
        help="Blend retriever order back into reranker output via rank fusion. 0 keeps pure reranker order.",
    )
    parser.add_argument(
        "--reranker-rank-fusion-k",
        type=int,
        default=60,
        help="Rank-fusion k used when blending retriever order into reranker output.",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=None,
        help="Hybrid weight for dense branch. Used only when --retriever hybrid.",
    )
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="",
        help="Comma-separated alphas for hybrid sweep, e.g. '0.0,0.1,0.2,0.5,1.0'.",
    )
    parser.add_argument(
        "--hybrid-rrf-k",
        type=int,
        default=None,
        help="RRF k value. Used only when --retriever hybrid.",
    )
    parser.add_argument(
        "--hybrid-candidate-k",
        type=int,
        default=None,
        help="Candidate pool size from each retriever. Used only when --retriever hybrid.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,nq,triviaqa",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Limit per dataset. Use <=0 to run full split.",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=None,
        help="Optional override qa.jsonl path. Supported only when running a single dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/runs"),
    )
    parser.add_argument(
        "--query-expansion-mode",
        choices=["off", "auto", "hyde", "hotpot_hyde", "hotpot_decompose"],
        default=None,
        help="Optional dense-query expansion mode applied before retrieval.",
    )
    parser.add_argument("--query-expansion-model", type=str, default=None)
    parser.add_argument(
        "--query-expansion-datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset allowlist for query expansion.",
    )
    parser.add_argument(
        "--query-expansion-max-completion-tokens",
        type=int,
        default=None,
        help="Completion budget for HyDE query expansion requests.",
    )
    parser.add_argument(
        "--continue-on-generation-error",
        action="store_true",
        help="Record generation errors in outputs and continue the run instead of aborting the batch.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write log output in addition to stderr.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Write heartbeat progress after every N processed queries. Use <=0 to disable count-based flushes.",
    )
    parser.add_argument(
        "--progress-min-seconds",
        type=float,
        default=60.0,
        help="Also write heartbeat progress when at least this many seconds elapsed since the last flush.",
    )
    return parser.parse_args()


def _recall_definition(dataset_name: str) -> str:
    if dataset_name == "hotpotqa":
        return "title_hit@k"
    return "answer_presence_proxy@k"


def _parse_alpha_grid(raw: str) -> list[float]:
    text = raw.strip()
    if not text:
        return []
    values: list[float] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        value = float(token)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"alpha must be in [0,1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("alpha-grid is empty after parsing")
    return values


def _alpha_to_label(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def _build_hybrid_retriever_from_loaded_branches(
    cfg: dict,
    alpha: float,
    bm25_retriever,
    dense_retriever,
) -> HybridRetriever:
    retrieval_cfg = cfg.get("retrieval", {})
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        alpha=float(alpha),
        rrf_k=int(retrieval_cfg.get("hybrid_rrf_k", 60)),
        candidate_k=int(retrieval_cfg.get("hybrid_candidate_k", 50)),
    )


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _unique_gold_title_count(gold_titles: list[str]) -> int:
    return len({normalize_title(title) for title in gold_titles if title.strip()})


def _all_gold_hit(gold_titles: list[str], hit_titles: list[str]) -> bool:
    gold_count = _unique_gold_title_count(gold_titles)
    if gold_count == 0:
        return False
    hit_count = len({normalize_title(title) for title in hit_titles if title.strip()})
    return hit_count >= gold_count


def _partial_metrics(results: list[RunExampleResult]) -> dict[str, object]:
    if not results:
        return {
            "EM": 0.0,
            "F1": 0.0,
            "Recall@k": 0.0,
            "AvgRetrievalLatencyMs": 0.0,
            "AvgRerankLatencyMs": 0.0,
            "AvgGenerationLatencyMs": 0.0,
            "AvgQueryExpansionLatencyMs": 0.0,
            "AvgExpandedQueriesPerSample": 0.0,
            "RecallAnyGoldTitle@k": 0.0,
            "RecallAllGold@k_title": 0.0,
            "RecallAllGold@raw_title": 0.0,
            "FailureBucketCounts": {},
            "NumQueryExpansionErrors": 0,
            "QueryExpansionErrorRate": 0.0,
        }

    hotpot_results = [result for result in results if _unique_gold_title_count(result.gold_titles) > 0]
    any_gold_hits = sum(
        1 for result in hotpot_results if any(title.strip() for title in result.gold_titles_in_final_top_k)
    )
    all_gold_hits = sum(
        1 for result in hotpot_results if _all_gold_hit(result.gold_titles, result.gold_titles_in_final_top_k)
    )
    raw_all_gold_hits = sum(
        1 for result in hotpot_results if _all_gold_hit(result.gold_titles, result.gold_titles_in_raw_candidates)
    )
    failure_bucket_counts = dict(
        Counter((result.retrieval_failure_bucket.strip() or "not_applicable") for result in hotpot_results)
    )
    query_expansion_errors = sum(1 for result in results if result.query_expansion_error.strip())
    query_expansion_fallbacks = sum(1 for result in results if result.query_expansion_used_fallback)

    return {
        "EM": _mean([1.0 if result.is_em else 0.0 for result in results]),
        "F1": _mean([result.f1 for result in results]),
        "Recall@k": _mean([result.recall_at_k for result in results]),
        "AvgRetrievalLatencyMs": _mean([result.retrieval_latency_ms for result in results]),
        "AvgRerankLatencyMs": _mean([result.rerank_latency_ms for result in results]),
        "AvgGenerationLatencyMs": _mean([result.generation_latency_ms for result in results]),
        "AvgQueryExpansionLatencyMs": _mean([result.query_expansion_latency_ms for result in results]),
        "AvgExpandedQueriesPerSample": _mean([float(len(result.expanded_queries)) for result in results]),
        "RecallAnyGoldTitle@k": (any_gold_hits / len(hotpot_results) if hotpot_results else 0.0),
        "RecallAllGold@k_title": (all_gold_hits / len(hotpot_results) if hotpot_results else 0.0),
        "RecallAllGold@raw_title": (raw_all_gold_hits / len(hotpot_results) if hotpot_results else 0.0),
        "FailureBucketCounts": failure_bucket_counts,
        "NumQueryExpansionErrors": query_expansion_errors,
        "QueryExpansionErrorRate": (query_expansion_errors / len(results) if results else 0.0),
        "NumQueryExpansionFallbacks": query_expansion_fallbacks,
        "QueryExpansionFallbackRate": (query_expansion_fallbacks / len(results) if results else 0.0),
    }


def _build_progress_payload(
    *,
    results: list[RunExampleResult],
    processed_queries: int,
    total_queries: int,
    started_at_iso: str,
    started_at_epoch: float,
    dataset_name: str,
    qa_path: Path,
    retriever_mode: str,
    query_expansion_mode: str,
    use_reranker: bool,
    status: str,
    error: str = "",
) -> dict[str, object]:
    now_iso = datetime.now().isoformat(timespec="seconds")
    elapsed_sec = max(0.0, time.time() - started_at_epoch)
    last_result = results[-1] if results else None
    payload = {
        "status": status,
        "dataset": dataset_name,
        "qa_path": str(qa_path.resolve()),
        "retriever": retriever_mode,
        "query_expansion_mode": query_expansion_mode,
        "use_reranker": bool(use_reranker),
        "started_at": started_at_iso,
        "updated_at": now_iso,
        "elapsed_sec": elapsed_sec,
        "processed_queries": processed_queries,
        "total_queries": total_queries,
        "progress_pct": ((processed_queries / total_queries) if total_queries else 0.0),
        "last_query_id": last_result.query_id if last_result is not None else "",
        "last_retrieval_failure_bucket": (
            last_result.retrieval_failure_bucket if last_result is not None else ""
        ),
        "last_query_expansion_failure_reason": (
            last_result.query_expansion_failure_reason if last_result is not None else ""
        ),
        "last_query_expansion_used_fallback": (
            bool(last_result.query_expansion_used_fallback) if last_result is not None else False
        ),
        "error": error,
        "metrics_snapshot": _partial_metrics(results),
    }
    return payload


def _make_progress_callback(
    *,
    dataset_dir: Path,
    dataset_name: str,
    qa_path: Path,
    retriever_mode: str,
    query_expansion_mode: str,
    use_reranker: bool,
    total_queries: int,
    progress_every: int,
    progress_min_seconds: float,
) -> tuple[
    Callable[[list[RunExampleResult], int, int], None],
    Callable[[list[RunExampleResult]], None],
    Callable[[str], None],
]:
    progress_path = dataset_dir / "progress.json"
    partial_predictions_path = dataset_dir / "predictions.partial.jsonl"
    if partial_predictions_path.exists():
        partial_predictions_path.unlink()

    started_at_iso = datetime.now().isoformat(timespec="seconds")
    started_at_epoch = time.time()
    save_json(
        progress_path,
        _build_progress_payload(
            results=[],
            processed_queries=0,
            total_queries=total_queries,
            started_at_iso=started_at_iso,
            started_at_epoch=started_at_epoch,
            dataset_name=dataset_name,
            qa_path=qa_path,
            retriever_mode=retriever_mode,
            query_expansion_mode=query_expansion_mode,
            use_reranker=use_reranker,
            status="running",
        ),
    )

    last_flush = time.monotonic()
    latest_results: list[RunExampleResult] = []
    latest_processed_queries = 0

    def progress_callback(results: list[RunExampleResult], processed_queries: int, total: int) -> None:
        nonlocal last_flush, latest_results, latest_processed_queries
        latest_results = results
        latest_processed_queries = processed_queries
        append_run_result_jsonl(partial_predictions_path, results[-1])
        due_by_count = progress_every > 0 and (processed_queries % progress_every == 0)
        due_by_time = progress_min_seconds > 0 and ((time.monotonic() - last_flush) >= progress_min_seconds)
        due_by_finish = processed_queries >= total
        if not (due_by_count or due_by_time or due_by_finish):
            return
        save_json(
            progress_path,
            _build_progress_payload(
                results=results,
                processed_queries=processed_queries,
                total_queries=total,
                started_at_iso=started_at_iso,
                started_at_epoch=started_at_epoch,
                dataset_name=dataset_name,
                qa_path=qa_path,
                retriever_mode=retriever_mode,
                query_expansion_mode=query_expansion_mode,
                use_reranker=use_reranker,
                status="running",
            ),
        )
        last_flush = time.monotonic()

    def write_completed(results: list[RunExampleResult]) -> None:
        save_json(
            progress_path,
            _build_progress_payload(
                results=results,
                processed_queries=len(results),
                total_queries=total_queries,
                started_at_iso=started_at_iso,
                started_at_epoch=started_at_epoch,
                dataset_name=dataset_name,
                qa_path=qa_path,
                retriever_mode=retriever_mode,
                query_expansion_mode=query_expansion_mode,
                use_reranker=use_reranker,
                status="completed",
            ),
        )

    def write_failed(error: str) -> None:
        save_json(
            progress_path,
            _build_progress_payload(
                results=latest_results,
                processed_queries=latest_processed_queries,
                total_queries=total_queries,
                started_at_iso=started_at_iso,
                started_at_epoch=started_at_epoch,
                dataset_name=dataset_name,
                qa_path=qa_path,
                retriever_mode=retriever_mode,
                query_expansion_mode=query_expansion_mode,
                use_reranker=use_reranker,
                status="failed",
                error=error,
            ),
        )

    return progress_callback, write_completed, write_failed


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, args.log_file)
    cfg = load_yaml_config(args.config)
    cfg.setdefault("retrieval", {})
    cfg.setdefault("generation", {})
    cfg.setdefault("query_expansion", {})
    if args.retriever is not None:
        cfg["retrieval"]["mode"] = args.retriever
    if args.top_k is not None:
        cfg["retrieval"]["top_k"] = int(args.top_k)
    retriever_mode = str(cfg["retrieval"].get("mode", "dense"))
    if args.dedup_mode is not None:
        cfg["retrieval"]["dedup_mode"] = args.dedup_mode
    if args.retrieve_top_k is not None:
        cfg["retrieval"]["retrieve_top_k"] = int(args.retrieve_top_k)
    if args.dedup_before_rerank:
        cfg["retrieval"]["dedup_before_rerank"] = True
    if args.title_first_rerank:
        cfg["retrieval"]["title_first_rerank"] = True
    if args.title_pool_k is not None:
        cfg["retrieval"]["title_pool_k"] = int(args.title_pool_k)
    if args.max_chunks_per_title is not None:
        cfg["retrieval"]["max_chunks_per_title"] = int(args.max_chunks_per_title)
    if args.min_unique_titles is not None:
        cfg["retrieval"]["min_unique_titles"] = int(args.min_unique_titles)
    if args.title_prefilter_manifest is not None:
        cfg["retrieval"]["title_prefilter_manifest_path"] = str(args.title_prefilter_manifest)
    if args.title_prefilter_bm25_path is not None:
        cfg["retrieval"]["title_prefilter_bm25_path"] = str(args.title_prefilter_bm25_path)
    if args.title_prefilter_docstore_path is not None:
        cfg["retrieval"]["title_prefilter_docstore_path"] = str(args.title_prefilter_docstore_path)
    if args.title_prefilter_k is not None:
        cfg["retrieval"]["title_prefilter_k"] = int(args.title_prefilter_k)
    if args.generator_mode is not None:
        cfg["generation"]["mode"] = args.generator_mode
    if args.generator_model is not None:
        cfg["generation"]["model"] = args.generator_model
    if args.generator_base_url is not None:
        cfg["generation"]["api_base"] = args.generator_base_url
    if args.generator_base_url_env is not None:
        cfg["generation"]["api_base_env"] = args.generator_base_url_env
    if args.generator_api_key_env is not None:
        cfg["generation"]["api_key_env"] = args.generator_api_key_env
    if args.generator_temperature is not None:
        cfg["generation"]["temperature"] = float(args.generator_temperature)
    if args.generator_max_output_tokens is not None:
        cfg["generation"]["max_output_tokens"] = int(args.generator_max_output_tokens)
    if args.generator_max_completion_tokens is not None:
        cfg["generation"]["max_completion_tokens"] = int(args.generator_max_completion_tokens)
    if args.generator_reasoning_split is not None:
        cfg["generation"]["reasoning_split"] = {
            "auto": "auto",
            "true": True,
            "false": False,
        }[args.generator_reasoning_split]
    if args.generator_timeout_sec is not None:
        cfg["generation"]["timeout_sec"] = int(args.generator_timeout_sec)
    if args.generator_input_price_per_1m is not None:
        cfg["generation"]["input_price_per_1m"] = float(args.generator_input_price_per_1m)
    if args.generator_output_price_per_1m is not None:
        cfg["generation"]["output_price_per_1m"] = float(args.generator_output_price_per_1m)
    if args.query_expansion_mode is not None:
        cfg["query_expansion"]["mode"] = args.query_expansion_mode
    if args.query_expansion_model is not None:
        cfg["query_expansion"]["model"] = args.query_expansion_model
    if args.query_expansion_datasets is not None:
        cfg["query_expansion"]["datasets"] = [
            token.strip() for token in args.query_expansion_datasets.split(",") if token.strip()
        ]
    if args.query_expansion_max_completion_tokens is not None:
        cfg["query_expansion"]["max_completion_tokens"] = int(args.query_expansion_max_completion_tokens)
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.qa_path is not None and len(selected_datasets) != 1:
        raise ValueError("--qa-path can only be used when --datasets selects exactly one dataset")
    configured_query_expansion_mode = str(cfg["query_expansion"].get("mode", "off")).strip().lower()

    if retriever_mode == "hybrid":
        if args.hybrid_rrf_k is not None:
            cfg["retrieval"]["hybrid_rrf_k"] = int(args.hybrid_rrf_k)
        if args.hybrid_candidate_k is not None:
            cfg["retrieval"]["hybrid_candidate_k"] = int(args.hybrid_candidate_k)
        alpha_values = _parse_alpha_grid(args.alpha_grid) if args.alpha_grid.strip() else []
        if not alpha_values and args.hybrid_alpha is not None:
            alpha_values = [float(args.hybrid_alpha)]
        if not alpha_values:
            alpha_values = [float(cfg["retrieval"].get("hybrid_alpha", 0.5))]
    else:
        alpha_values = [None]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _evalops_client = EvalOpsClient.from_env()
    run_name = f"naive_baseline_{retriever_mode}_{run_id}"
    is_hybrid_sweep = retriever_mode == "hybrid" and len(alpha_values) > 1
    if is_hybrid_sweep:
        run_name = f"naive_baseline_hybrid_sweep_{run_id}"
    if args.use_reranker:
        run_name = f"{run_name}_rerank"
    out_dir = ensure_dir(args.output_root / run_name)

    run_cfg = {
        "retriever": retriever_mode,
        "top_k": int(cfg["retrieval"].get("top_k", 5)),
        "generator_mode": cfg["generation"].get("mode", "extractive"),
        "generator_model": cfg["generation"].get("model", ""),
        "generator_api_base": cfg["generation"].get("api_base", ""),
        "generator_api_base_env": cfg["generation"].get("api_base_env", ""),
        "generator_api_key_env": cfg["generation"].get("api_key_env", ""),
        "generator_temperature": cfg["generation"].get("temperature"),
        "generator_max_output_tokens": cfg["generation"].get("max_output_tokens"),
        "generator_max_completion_tokens": cfg["generation"].get("max_completion_tokens"),
        "generator_reasoning_split": cfg["generation"].get("reasoning_split", "auto"),
        "generator_timeout_sec": cfg["generation"].get("timeout_sec"),
        "generator_input_price_per_1m": cfg["generation"].get("input_price_per_1m", 0.0),
        "generator_output_price_per_1m": cfg["generation"].get("output_price_per_1m", 0.0),
        "use_reranker": bool(args.use_reranker),
        "retrieve_top_k": int(cfg["retrieval"].get("retrieve_top_k", args.retrieve_top_k or 20)),
        "dedup_mode": cfg["retrieval"].get("dedup_mode", "off"),
        "dedup_before_rerank": bool(cfg["retrieval"].get("dedup_before_rerank", False)),
        "title_first_rerank": bool(cfg["retrieval"].get("title_first_rerank", False)),
        "title_pool_k": int(cfg["retrieval"].get("title_pool_k", 40)),
        "max_chunks_per_title": int(cfg["retrieval"].get("max_chunks_per_title", 2)),
        "min_unique_titles": int(cfg["retrieval"].get("min_unique_titles", 6)),
        "title_prefilter_manifest_path": cfg["retrieval"].get("title_prefilter_manifest_path", ""),
        "title_prefilter_bm25_path": cfg["retrieval"].get("title_prefilter_bm25_path", ""),
        "title_prefilter_docstore_path": cfg["retrieval"].get("title_prefilter_docstore_path", ""),
        "title_prefilter_k": cfg["retrieval"].get("title_prefilter_k"),
        "query_expansion_mode": configured_query_expansion_mode,
        "query_expansion_datasets": cfg["query_expansion"].get("datasets", []),
        "query_expansion_model": cfg["query_expansion"].get("model", "") or cfg["generation"].get("model", ""),
        "query_expansion_api_base": cfg["query_expansion"].get("api_base", ""),
        "query_expansion_api_base_env": (
            cfg["query_expansion"].get("api_base_env", "") or cfg["generation"].get("api_base_env", "")
        ),
        "query_expansion_api_key_env": (
            cfg["query_expansion"].get("api_key_env", "") or cfg["generation"].get("api_key_env", "")
        ),
        "query_expansion_temperature": cfg["query_expansion"].get("temperature", 0.0),
        "query_expansion_max_completion_tokens": cfg["query_expansion"].get(
            "max_completion_tokens",
            512 if configured_query_expansion_mode == "hotpot_decompose" else 256,
        ),
        "query_expansion_include_original_query": cfg["query_expansion"].get("include_original_query", True),
        "query_expansion_max_queries": cfg["query_expansion"].get("max_queries", 3),
        "query_expansion_timeout_sec": cfg["query_expansion"].get(
            "timeout_sec",
            cfg["generation"].get("timeout_sec"),
        ),
        "reranker_model": args.reranker_model,
        "reranker_batch_size": int(args.reranker_batch_size),
        "reranker_retriever_rank_weight": float(args.reranker_retriever_rank_weight),
        "reranker_rank_fusion_k": int(args.reranker_rank_fusion_k),
        "datasets": selected_datasets,
        "qa_path_override": str(args.qa_path.resolve()) if args.qa_path is not None else "",
        "max_queries": args.max_queries,
        "continue_on_generation_error": bool(args.continue_on_generation_error),
        "progress_every": int(args.progress_every),
        "progress_min_seconds": float(args.progress_min_seconds),
        "alpha_values": [a for a in alpha_values if a is not None],
        "hybrid_rrf_k": cfg["retrieval"].get("hybrid_rrf_k"),
        "hybrid_candidate_k": cfg["retrieval"].get("hybrid_candidate_k"),
    }
    run_cfg["query_expansion_effective_modes_by_dataset"] = {
        dataset_name: resolve_query_expansion_mode(cfg, dataset_name)
        for dataset_name in selected_datasets
    }

    all_metrics: dict[str, dict] = {}
    reranker = None
    if args.use_reranker:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            batch_size=args.reranker_batch_size,
            retriever_rank_weight=float(args.reranker_retriever_rank_weight),
            rank_fusion_k=int(args.reranker_rank_fusion_k),
        )
    generator = build_generator(cfg)
    if hasattr(generator, "reasoning_split"):
        run_cfg["generator_effective_reasoning_split"] = bool(generator.reasoning_split)
    if hasattr(generator, "max_completion_tokens"):
        run_cfg["generator_effective_max_completion_tokens"] = generator.max_completion_tokens
    first_active_query_expander = None
    for dataset_name in selected_datasets:
        first_active_query_expander = build_query_expander(cfg, dataset_name=dataset_name)
        if first_active_query_expander is not None:
            break
    if first_active_query_expander is not None and hasattr(first_active_query_expander, "model"):
        run_cfg["query_expansion_effective_model"] = first_active_query_expander.model
    if first_active_query_expander is not None and hasattr(first_active_query_expander, "max_completion_tokens"):
        run_cfg["query_expansion_effective_max_completion_tokens"] = first_active_query_expander.max_completion_tokens
    if first_active_query_expander is not None and hasattr(first_active_query_expander, "include_original_query"):
        run_cfg["query_expansion_effective_include_original_query"] = (
            first_active_query_expander.include_original_query
        )
    if first_active_query_expander is not None and hasattr(first_active_query_expander, "max_queries"):
        run_cfg["query_expansion_effective_max_queries"] = first_active_query_expander.max_queries
    save_json(out_dir / "run_config.json", run_cfg)

    bm25_retriever = None
    dense_retriever = None
    run_cfg["retriever"] = retriever_mode
    if retriever_mode == "hybrid":
        bm25_cfg = dict(cfg)
        bm25_cfg["retrieval"] = dict(cfg["retrieval"])
        bm25_cfg["retrieval"]["mode"] = "bm25"
        dense_cfg = dict(cfg)
        dense_cfg["retrieval"] = dict(cfg["retrieval"])
        dense_cfg["retrieval"]["mode"] = "dense"
        bm25_retriever = build_retriever(bm25_cfg, corpus=[])
        dense_retriever = build_retriever(dense_cfg, corpus=[])

    for alpha in alpha_values:
        if alpha is not None:
            cfg["retrieval"]["hybrid_alpha"] = float(alpha)
            if is_hybrid_sweep:
                alpha_label = _alpha_to_label(alpha)
                alpha_dir = ensure_dir(out_dir / alpha_label)
                all_metrics[alpha_label] = {}
            else:
                alpha_label = ""
                alpha_dir = out_dir
            print(f"Running hybrid with alpha={alpha:.2f}")
        else:
            alpha_label = ""
            alpha_dir = out_dir

        if retriever_mode == "hybrid":
            retriever = _build_hybrid_retriever_from_loaded_branches(
                cfg=cfg,
                alpha=float(alpha),
                bm25_retriever=bm25_retriever,
                dense_retriever=dense_retriever,
            )
        else:
            retriever = build_retriever(cfg, corpus=[])
        for dataset_name in selected_datasets:
            if dataset_name not in DATASET_PATHS:
                print(f"Skip unknown dataset: {dataset_name}")
                continue
            effective_query_expansion_mode = resolve_query_expansion_mode(cfg, dataset_name)
            query_expander = build_query_expander(
                cfg,
                dataset_name=dataset_name,
                mode_override=effective_query_expansion_mode,
            )

            qa_path = args.qa_path if args.qa_path is not None else DATASET_PATHS[dataset_name]
            max_q = args.max_queries if args.max_queries > 0 else None
            samples = load_flashrag_qa(qa_path, max_queries=max_q)
            print(f"Running {dataset_name}: {len(samples)} queries")
            dataset_dir = ensure_dir(alpha_dir / dataset_name)
            progress_callback, write_progress_completed, write_progress_failed = _make_progress_callback(
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                qa_path=qa_path,
                retriever_mode=retriever_mode,
                query_expansion_mode=effective_query_expansion_mode,
                use_reranker=bool(args.use_reranker),
                total_queries=len(samples),
                progress_every=int(args.progress_every),
                progress_min_seconds=float(args.progress_min_seconds),
            )
            try:
                results, metrics = run_naive_rag(
                    retriever,
                    samples,
                    top_k=int(cfg["retrieval"].get("top_k", 5)),
                    generator=generator,
                    reranker=reranker,
                    retrieve_top_k=int(cfg["retrieval"].get("retrieve_top_k", args.retrieve_top_k or 20)),
                    dedup_mode=str(cfg["retrieval"].get("dedup_mode", "off")),
                    dedup_before_rerank=bool(cfg["retrieval"].get("dedup_before_rerank", False)),
                    title_first_rerank=bool(cfg["retrieval"].get("title_first_rerank", False)),
                    title_pool_k=int(cfg["retrieval"].get("title_pool_k", 40)),
                    max_chunks_per_title=int(cfg["retrieval"].get("max_chunks_per_title", 2)),
                    min_unique_titles=int(cfg["retrieval"].get("min_unique_titles", 6)),
                    query_expander=query_expander,
                    query_expansion_mode=effective_query_expansion_mode,
                    continue_on_generation_error=bool(args.continue_on_generation_error),
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                write_progress_failed(f"{type(exc).__name__}: {exc}")
                raise
            metrics["Dataset"] = dataset_name
            metrics["NumQueries"] = len(samples)
            metrics["Retriever"] = retriever_mode
            metrics["GeneratorMode"] = cfg["generation"].get("mode", "extractive")
            metrics["GeneratorModel"] = cfg["generation"].get("model", "")
            metrics["UseReranker"] = bool(args.use_reranker)
            metrics["DedupMode"] = str(cfg["retrieval"].get("dedup_mode", "off"))
            metrics["DedupBeforeRerank"] = bool(cfg["retrieval"].get("dedup_before_rerank", False))
            metrics["RetrieveTopK"] = int(cfg["retrieval"].get("retrieve_top_k", args.retrieve_top_k or 20))
            metrics["TitleFirstRerank"] = bool(cfg["retrieval"].get("title_first_rerank", False))
            metrics["TitlePoolK"] = int(cfg["retrieval"].get("title_pool_k", 40))
            metrics["MaxChunksPerTitle"] = int(cfg["retrieval"].get("max_chunks_per_title", 2))
            metrics["MinUniqueTitles"] = int(cfg["retrieval"].get("min_unique_titles", 6))
            if retriever_mode == "dense_sharded_title_prefilter":
                metrics["TitlePrefilterManifestPath"] = cfg["retrieval"].get("title_prefilter_manifest_path", "")
                metrics["TitlePrefilterBm25Path"] = cfg["retrieval"].get("title_prefilter_bm25_path", "")
                metrics["TitlePrefilterDocstorePath"] = cfg["retrieval"].get("title_prefilter_docstore_path", "")
                metrics["TitlePrefilterK"] = int(cfg["retrieval"].get("title_prefilter_k", 30))
            metrics["ConfiguredQueryExpansionMode"] = configured_query_expansion_mode
            metrics["QueryExpansionMode"] = effective_query_expansion_mode
            metrics["ContinueOnGenerationError"] = bool(args.continue_on_generation_error)
            metrics["QAPath"] = str(qa_path.resolve())
            if args.use_reranker:
                metrics["RerankerModel"] = args.reranker_model
                metrics["RerankerRetrieverRankWeight"] = float(args.reranker_retriever_rank_weight)
                metrics["RerankerRankFusionK"] = int(args.reranker_rank_fusion_k)
            metrics["RecallDefinition"] = _recall_definition(dataset_name)
            if alpha is not None:
                metrics["HybridAlpha"] = float(alpha)
                metrics["HybridRrfK"] = int(cfg["retrieval"].get("hybrid_rrf_k", 60))
                metrics["HybridCandidateK"] = int(cfg["retrieval"].get("hybrid_candidate_k", 50))
                if is_hybrid_sweep:
                    all_metrics[alpha_label][dataset_name] = metrics
                else:
                    all_metrics[dataset_name] = metrics
            else:
                all_metrics[dataset_name] = metrics

            save_json(dataset_dir / "metrics.json", metrics)
            # Stamp run_id for per-example traceability before serializing
            for r in results:
                r.run_id = out_dir.name
            save_run_results(dataset_dir / "predictions.json", results)
            write_progress_completed(results)
            # EvalOps: submit run report (fails silently if endpoint not configured)
            _evalops_client.submit(
                build_eval_run_report(
                    out_dir.name,
                    metrics,
                    results,
                    dataset=dataset_name,
                    retriever_mode=retriever_mode,
                )
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))

    save_json(out_dir / "summary_metrics.json", all_metrics)
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
