from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    page_start: int | None = None   # first page this chunk spans (1-indexed)
    page_end: int | None = None     # last page; equals page_start for single-page chunks
    section: str | None = None      # heuristic heading / section label
    source: str | None = None       # origin filename or URL
    extra_metadata: dict = field(default_factory=dict)


@dataclass
class ScoredDocument:
    """API-layer wrapper that pairs a Document with its retrieval score.

    Only used in ``src/api/``; the internal retrieval pipeline continues to
    operate on plain ``Document`` lists so existing code paths are unchanged.
    """

    document: Document
    score: float          # normalized similarity or rerank score
    retrieval_stage: str  # "dense" | "bm25" | "rerank"


@dataclass(frozen=True)
class QuerySample:
    query_id: str
    question: str
    answers: list[str]
    gold_doc_id: str | None = None
    gold_titles: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query_id: str
    retrieved_doc_ids: list[str]


@dataclass
class RunExampleResult:
    query_id: str
    predicted_answer: str
    gold_answers: list[str]
    retrieved_doc_ids: list[str]
    retrieved_titles: list[str]
    unique_retrieved_titles: int
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    approx_input_tokens: int
    approx_output_tokens: int
    is_em: bool
    f1: float
    recall_at_k: float
    raw_candidate_count: int
    dedup_candidate_count: int
    duplicate_candidates_removed: int
    dedup_mode: str = "off"
    expanded_query: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    query_expansion_mode: str = "off"
    query_expansion_latency_ms: float = 0.0
    query_expansion_error: str = ""
    query_expansion_failure_reason: str = ""
    query_expansion_cache_key: str = ""
    query_expansion_used_fallback: bool = False
    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    actual_reasoning_tokens: int | None = None
    generation_cost_usd: float | None = None
    generation_provider: str = ""
    generation_model: str = ""
    generation_error: str = ""
    gold_titles: list[str] = field(default_factory=list)
    gold_title_ranks: dict[str, int | None] = field(default_factory=dict)
    gold_titles_in_raw_candidates: list[str] = field(default_factory=list)
    gold_titles_after_dedup: list[str] = field(default_factory=list)
    gold_titles_in_final_top_k: list[str] = field(default_factory=list)
    missing_gold_count: int = 0
    first_gold_found: bool = False
    second_gold_found: bool = False
    retrieval_failure_bucket: str = ""
    retrieved_texts: list[str] = field(default_factory=list)
    question: str = ""
    # A3 NLI citation metrics (None = not computed; only set for PDF Q&A datasets)
    answer_attribution_rate: float | None = None
    supporting_passage_hit: float | None = None
    page_grounding_accuracy: float | None = None
    # C2 bad-case traceability fields
    failure_stage: str = ""   # e.g. "retrieval", "generation", "rerank"
    failure_detail: str = ""  # free-text error description
    run_id: str = ""          # identifier of the originating eval run
