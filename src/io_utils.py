import json
from pathlib import Path
from typing import Any

from src.types import RunExampleResult


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_result_to_dict(r: RunExampleResult) -> dict[str, Any]:
    return {
        "query_id": r.query_id,
        "question": r.question,
        "predicted_answer": r.predicted_answer,
        "gold_answers": r.gold_answers,
        "retrieved_doc_ids": r.retrieved_doc_ids,
        "retrieved_titles": r.retrieved_titles,
        "retrieved_texts": r.retrieved_texts,
        "unique_retrieved_titles": r.unique_retrieved_titles,
        "retrieval_latency_ms": r.retrieval_latency_ms,
        "rerank_latency_ms": r.rerank_latency_ms,
        "generation_latency_ms": r.generation_latency_ms,
        "approx_input_tokens": r.approx_input_tokens,
        "approx_output_tokens": r.approx_output_tokens,
        "raw_candidate_count": r.raw_candidate_count,
        "dedup_candidate_count": r.dedup_candidate_count,
        "duplicate_candidates_removed": r.duplicate_candidates_removed,
        "dedup_mode": r.dedup_mode,
        "expanded_query": r.expanded_query,
        "expanded_queries": r.expanded_queries,
        "query_expansion_mode": r.query_expansion_mode,
        "query_expansion_latency_ms": r.query_expansion_latency_ms,
        "query_expansion_error": r.query_expansion_error,
        "query_expansion_failure_reason": r.query_expansion_failure_reason,
        "query_expansion_cache_key": r.query_expansion_cache_key,
        "query_expansion_used_fallback": r.query_expansion_used_fallback,
        "actual_input_tokens": r.actual_input_tokens,
        "actual_output_tokens": r.actual_output_tokens,
        "actual_reasoning_tokens": r.actual_reasoning_tokens,
        "generation_cost_usd": r.generation_cost_usd,
        "generation_provider": r.generation_provider,
        "generation_model": r.generation_model,
        "generation_error": r.generation_error,
        "is_em": r.is_em,
        "f1": r.f1,
        "recall_at_k": r.recall_at_k,
        "gold_titles": r.gold_titles,
        "gold_title_ranks": r.gold_title_ranks,
        "gold_titles_in_raw_candidates": r.gold_titles_in_raw_candidates,
        "gold_titles_after_dedup": r.gold_titles_after_dedup,
        "gold_titles_in_final_top_k": r.gold_titles_in_final_top_k,
        "missing_gold_count": r.missing_gold_count,
        "first_gold_found": r.first_gold_found,
        "second_gold_found": r.second_gold_found,
        "retrieval_failure_bucket": r.retrieval_failure_bucket,
        # A3 NLI citation metrics
        "answer_attribution_rate": r.answer_attribution_rate,
        "supporting_passage_hit": r.supporting_passage_hit,
        "page_grounding_accuracy": r.page_grounding_accuracy,
        # C2 bad-case traceability
        "failure_stage": r.failure_stage,
        "failure_detail": r.failure_detail,
        "run_id": r.run_id,
        # Phase 5 generation quality
        "citation_count": r.citation_count,
        "citation_precision": r.citation_precision,
        "hedging_detected": r.hedging_detected,
    }


def append_run_result_jsonl(path: str | Path, row: RunExampleResult) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_result_to_dict(row), ensure_ascii=False) + "\n")


def save_run_results(path: str | Path, rows: list[RunExampleResult]) -> None:
    payload = [run_result_to_dict(r) for r in rows]
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
