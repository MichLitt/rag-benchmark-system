"""EvalRunReport — versioned schema for a single RAG evaluation run.

schema_version is always ``"rag/v1"`` so downstream consumers can gate on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalRunReport:
    """Versioned, self-describing record of one eval run.

    Produced by :func:`src.evalops.adapter.build_eval_run_report` and
    submitted (fire-and-forget) via :class:`src.evalops.client.EvalOpsClient`.

    Fields
    ------
    schema_version
        Always ``"rag/v1"``.  Bump the minor component for additive changes,
        major component for breaking changes.
    run_id
        Opaque identifier propagated from the orchestration layer (e.g. the
        timestamped output directory name).
    retrieval_profile
        One lightweight dict per processed example, carrying per-example
        latency, failure bucket, and C2 traceability fields.  Kept as plain
        dicts (not a typed dataclass) so the schema stays forward-compatible
        with arbitrary extra keys written by future pipeline stages.
    """

    schema_version: str = "rag/v1"
    run_id: str = ""
    dataset: str = ""
    retriever_mode: str = ""
    generator_model: str = ""
    num_queries: int = 0

    # Aggregate QA metrics
    em: float = 0.0
    f1: float = 0.0
    recall_at_k: float = 0.0

    # Faithfulness (populated post-hoc by score_faithfulness.py)
    avg_faithfulness: float | None = None
    hallucination_rate: float | None = None

    # Latency breakdown (milliseconds)
    avg_retrieval_latency_ms: float = 0.0
    avg_rerank_latency_ms: float = 0.0
    avg_generation_latency_ms: float = 0.0
    avg_query_expansion_latency_ms: float = 0.0

    # Cost
    total_generation_cost_usd: float | None = None
    avg_generation_cost_usd: float | None = None

    # Per-example retrieval tracing
    retrieval_profile: list[dict] = field(default_factory=list)
