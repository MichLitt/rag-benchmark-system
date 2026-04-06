"""Adapter: convert pipeline outputs into an EvalRunReport.

The adapter is the only place that knows about both the internal
``RunExampleResult`` dataclass and the ``EvalRunReport`` schema, keeping both
sides decoupled.
"""
from __future__ import annotations

from src.evalops.schema import EvalRunReport
from src.types import RunExampleResult


def build_eval_run_report(
    run_id: str,
    metrics: dict,
    results: list[RunExampleResult],
    *,
    dataset: str = "",
    retriever_mode: str = "",
    generator_model: str = "",
) -> EvalRunReport:
    """Build an :class:`EvalRunReport` from orchestration-layer outputs.

    Args:
        run_id: Opaque run identifier (e.g. timestamped directory name).
        metrics: Aggregated metrics dict as returned by ``run_naive_rag``.
        results: Per-example results from the same run.
        dataset: Dataset name override (falls back to ``metrics["Dataset"]``).
        retriever_mode: Retriever mode override (falls back to ``metrics["Retriever"]``).
        generator_model: Generator model override (falls back to ``metrics["GeneratorModel"]``).

    Returns:
        A fully populated :class:`EvalRunReport`.
    """
    # Cost aggregation — only over examples that have a real cost value
    costs = [r.generation_cost_usd for r in results if r.generation_cost_usd is not None]
    total_cost: float | None = sum(costs) if costs else None
    avg_cost: float | None = (total_cost / len(costs)) if costs else None

    # Per-example retrieval profile for tracing / debugging
    profile = [
        {
            "query_id": r.query_id,
            "retrieval_latency_ms": r.retrieval_latency_ms,
            "rerank_latency_ms": r.rerank_latency_ms,
            "generation_latency_ms": r.generation_latency_ms,
            "retrieval_failure_bucket": r.retrieval_failure_bucket,
            "failure_stage": r.failure_stage,
            "failure_detail": r.failure_detail,
        }
        for r in results
    ]

    return EvalRunReport(
        run_id=run_id,
        dataset=dataset or str(metrics.get("Dataset", "")),
        retriever_mode=retriever_mode or str(metrics.get("Retriever", "")),
        generator_model=generator_model or str(metrics.get("GeneratorModel", "")),
        num_queries=int(metrics.get("NumQueries", len(results))),
        em=float(metrics.get("EM", 0.0)),
        f1=float(metrics.get("F1", 0.0)),
        recall_at_k=float(metrics.get("Recall@k", 0.0)),
        avg_retrieval_latency_ms=float(metrics.get("AvgRetrievalLatencyMs", 0.0)),
        avg_rerank_latency_ms=float(metrics.get("AvgRerankLatencyMs", 0.0)),
        avg_generation_latency_ms=float(metrics.get("AvgGenerationLatencyMs", 0.0)),
        avg_query_expansion_latency_ms=float(metrics.get("AvgQueryExpansionLatencyMs", 0.0)),
        total_generation_cost_usd=total_cost,
        avg_generation_cost_usd=avg_cost,
        retrieval_profile=profile,
    )
