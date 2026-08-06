import time
from statistics import mean
from typing import Callable, Protocol

import tiktoken

from src.logging_utils import get_logger

logger = get_logger(__name__)

from src.evaluation.metrics import exact_match_any, max_f1_score, recall_at_k
from src.generation.base import GeneratorLike
from src.generation.extractive import ExtractiveGenerator
from src.query.base import QueryExpanderLike
from src.retrieval.postprocess import (
    build_hotpot_gold_diagnostics,
    build_retrieval_diagnostics,
    deduplicate_documents,
    pack_title_diverse_documents,
    select_title_representatives,
)
from src.types import Document, QuerySample, RunExampleResult


class RetrieverLike(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        ...

    def retrieve_many(self, queries: list[str], top_k: int) -> list[list[Document]]:
        ...


class RerankerLike(Protocol):
    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        ...


def _document_key(doc: Document) -> str:
    key = doc.doc_id.strip()
    if key:
        return key
    title = doc.title.strip()
    if title:
        return title
    return doc.text.strip()


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(_normalize(text)))


def _answer_presence_recall_at_k(docs: list[Document], answers: list[str], k: int) -> float:
    if not answers:
        return 0.0
    answer_norm = [_normalize(a) for a in answers if a.strip()]
    if not answer_norm:
        return 0.0
    top_docs = docs[:k]
    for doc in top_docs:
        hay = _normalize(f"{doc.title} {doc.text}")
        if any(a and a in hay for a in answer_norm):
            return 1.0
    return 0.0


def _recall_for_sample(sample: QuerySample, docs: list[Document], top_k: int) -> float:
    doc_ids = [d.doc_id for d in docs]
    if sample.gold_doc_id:
        return recall_at_k(doc_ids, sample.gold_doc_id, top_k)

    if sample.gold_titles:
        top_titles = {_normalize(d.title) for d in docs[:top_k] if d.title}
        gold_titles = {_normalize(t) for t in sample.gold_titles if t}
        return 1.0 if top_titles.intersection(gold_titles) else 0.0

    return _answer_presence_recall_at_k(docs, sample.answers, top_k)


def _unique_gold_title_count(gold_titles: list[str]) -> int:
    return len({_normalize(title) for title in gold_titles if title.strip()})


def _has_any_gold_title_hit(hit_titles: list[str]) -> bool:
    return any(title.strip() for title in hit_titles)


def _has_all_gold_title_hits(gold_titles: list[str], hit_titles: list[str]) -> bool:
    gold_count = _unique_gold_title_count(gold_titles)
    if gold_count == 0:
        return False
    hit_count = len({_normalize(title) for title in hit_titles if title.strip()})
    return hit_count >= gold_count


def _hotpot_metrics(results: list[RunExampleResult]) -> dict[str, float | dict[str, int]]:
    hotpot_results = [result for result in results if _unique_gold_title_count(result.gold_titles) > 0]
    if not hotpot_results:
        return {}

    total = len(hotpot_results)
    any_gold_hits = sum(
        1 for result in hotpot_results if _has_any_gold_title_hit(result.gold_titles_in_final_top_k)
    )
    all_gold_hits = sum(
        1
        for result in hotpot_results
        if _has_all_gold_title_hits(result.gold_titles, result.gold_titles_in_final_top_k)
    )
    raw_all_gold_hits = sum(
        1
        for result in hotpot_results
        if _has_all_gold_title_hits(result.gold_titles, result.gold_titles_in_raw_candidates)
    )

    failure_bucket_counts: dict[str, int] = {}
    for result in hotpot_results:
        bucket = result.retrieval_failure_bucket.strip() or "not_applicable"
        failure_bucket_counts[bucket] = failure_bucket_counts.get(bucket, 0) + 1

    return {
        "RecallAnyGoldTitle@k": (any_gold_hits / total if total else 0.0),
        "RecallAllGold@k_title": (all_gold_hits / total if total else 0.0),
        "RecallAllGold@raw_title": (raw_all_gold_hits / total if total else 0.0),
        "FailureBucketCounts": failure_bucket_counts,
    }


def _normalize_expanded_queries(queries: list[str], fallback_question: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(query.split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    if normalized:
        return normalized
    return [" ".join(fallback_question.split()).strip()]


def _resolve_expanded_queries(
    question: str,
    query_expander: QueryExpanderLike | None,
) -> list[str]:
    if query_expander is None:
        return [" ".join(question.split()).strip()]

    if hasattr(query_expander, "expand_queries"):
        raw_queries = query_expander.expand_queries(question)
        return _normalize_expanded_queries(list(raw_queries), question)

    expanded_query = query_expander.expand(question).strip()
    if not expanded_query:
        raise RuntimeError("Query expander returned an empty query.")
    return [expanded_query]


def _query_expansion_metadata(query_expander: QueryExpanderLike | None) -> dict[str, object]:
    if query_expander is None:
        return {}
    getter = getattr(query_expander, "get_last_expansion_metadata", None)
    if callable(getter):
        payload = getter()
        if isinstance(payload, dict):
            return payload
    return {}


def _fuse_multi_query_results(
    retriever: RetrieverLike,
    queries: list[str],
    top_k: int,
    rank_fusion_k: int = 60,
) -> list[Document]:
    if len(queries) == 1:
        return retriever.retrieve(queries[0], top_k=top_k)

    if hasattr(retriever, "retrieve_many"):
        batched_docs = retriever.retrieve_many(queries, top_k=top_k)
    else:
        batched_docs = [retriever.retrieve(query, top_k=top_k) for query in queries]

    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for docs in batched_docs:
        for rank, doc in enumerate(docs, start=1):
            key = _document_key(doc)
            if not key:
                continue
            by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 / (rank_fusion_k + rank))

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [by_key[key] for key in ranked_keys]


def run_naive_rag(
    retriever: RetrieverLike,
    eval_set: list[QuerySample],
    top_k: int = 5,
    generator: GeneratorLike | None = None,
    reranker: RerankerLike | None = None,
    retrieve_top_k: int | None = None,
    dedup_mode: str = "off",
    dedup_before_rerank: bool = False,
    title_first_rerank: bool = False,
    title_pool_k: int = 40,
    max_chunks_per_title: int = 2,
    min_unique_titles: int = 6,
    query_expander: QueryExpanderLike | None = None,
    query_expansion_mode: str = "off",
    continue_on_generation_error: bool = False,
    progress_callback: Callable[[list[RunExampleResult], int, int], None] | None = None,
    postprocess_answers: bool = False,
) -> tuple[list[RunExampleResult], dict[str, float]]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if retrieve_top_k is not None and retrieve_top_k <= 0:
        raise ValueError(f"retrieve_top_k must be > 0, got {retrieve_top_k}")
    if title_pool_k <= 0:
        raise ValueError(f"title_pool_k must be > 0, got {title_pool_k}")
    if max_chunks_per_title <= 0:
        raise ValueError(f"max_chunks_per_title must be > 0, got {max_chunks_per_title}")
    if min_unique_titles < 0:
        raise ValueError(f"min_unique_titles must be >= 0, got {min_unique_titles}")

    generator = generator or ExtractiveGenerator()

    results: list[RunExampleResult] = []
    retrieval_latencies_ms: list[float] = []
    rerank_latencies_ms: list[float] = []
    generation_latencies_ms: list[float] = []
    input_token_counts: list[int] = []
    output_token_counts: list[int] = []
    actual_input_token_counts: list[int] = []
    actual_output_token_counts: list[int] = []
    actual_reasoning_token_counts: list[int] = []
    generation_costs_usd: list[float] = []
    unique_title_counts: list[int] = []
    raw_candidate_counts: list[int] = []
    dedup_candidate_counts: list[int] = []
    duplicate_candidates_removed_counts: list[int] = []
    expanded_query_counts: list[int] = []
    query_expansion_latencies_ms: list[float] = []
    generation_errors = 0
    query_expansion_errors = 0
    retrieval_k = max(top_k, retrieve_top_k or top_k)
    total_samples = len(eval_set)
    for sample in eval_set:
        expanded_queries = [sample.question]
        expanded_query = sample.question
        query_expansion_error = ""
        query_expansion_failure_reason = ""
        query_expansion_cache_key = ""
        query_expansion_used_fallback = False
        query_expansion_latency_ms = 0.0
        if query_expander is not None:
            start_query_expansion = time.perf_counter()
            try:
                expanded_queries = _resolve_expanded_queries(sample.question, query_expander)
                expanded_query = " || ".join(expanded_queries)
                expansion_meta = _query_expansion_metadata(query_expander)
                query_expansion_failure_reason = str(expansion_meta.get("failure_reason", "")).strip()
                query_expansion_cache_key = str(expansion_meta.get("cache_key", "")).strip()
                query_expansion_used_fallback = bool(expansion_meta.get("used_fallback", False))
            except Exception as exc:
                logger.warning("Query expansion failed for %s: %s", sample.query_id, exc)
                query_expansion_errors += 1
                query_expansion_error = f"{type(exc).__name__}: {exc}"
                expansion_meta = _query_expansion_metadata(query_expander)
                query_expansion_failure_reason = str(expansion_meta.get("failure_reason", "")).strip()
                query_expansion_cache_key = str(expansion_meta.get("cache_key", "")).strip()
                query_expansion_used_fallback = bool(expansion_meta.get("used_fallback", False))
                expanded_queries = [sample.question]
                expanded_query = sample.question
            query_expansion_latency_ms = (time.perf_counter() - start_query_expansion) * 1000
        else:
            expanded_queries = [sample.question]
            expanded_query = sample.question
        query_expansion_latencies_ms.append(query_expansion_latency_ms)
        expanded_query_counts.append(len(expanded_queries))

        start_retrieval = time.perf_counter()
        raw_candidate_docs = _fuse_multi_query_results(
            retriever,
            expanded_queries,
            top_k=retrieval_k,
            rank_fusion_k=int(getattr(query_expander, "rank_fusion_k", 60)),
        )
        retrieval_latency_ms = (time.perf_counter() - start_retrieval) * 1000
        retrieval_latencies_ms.append(retrieval_latency_ms)

        if dedup_mode != "off" and dedup_before_rerank:
            candidate_docs = deduplicate_documents(raw_candidate_docs, dedup_mode)
        else:
            candidate_docs = list(raw_candidate_docs)

        if title_first_rerank:
            candidate_docs = select_title_representatives(candidate_docs, max_titles=title_pool_k)

        rerank_latency_ms = 0.0
        if reranker is not None:
            start_rerank = time.perf_counter()
            ranked_docs = reranker.rerank(sample.question, candidate_docs, top_k=len(candidate_docs))
            rerank_latency_ms = (time.perf_counter() - start_rerank) * 1000
        else:
            ranked_docs = list(candidate_docs)

        if dedup_mode != "off" and not dedup_before_rerank:
            deduped_docs = deduplicate_documents(ranked_docs, dedup_mode)
        else:
            deduped_docs = list(ranked_docs)

        if title_first_rerank:
            docs = pack_title_diverse_documents(
                ranked_title_docs=ranked_docs,
                raw_candidate_docs=raw_candidate_docs,
                top_k=top_k,
                max_chunks_per_title=max_chunks_per_title,
                min_unique_titles=min_unique_titles,
            )
        else:
            docs = deduped_docs[:top_k]
        retrieval_diagnostics = build_retrieval_diagnostics(
            raw_candidates=raw_candidate_docs,
            deduped_candidates=deduped_docs,
            final_docs=docs,
        )
        hotpot_gold_diagnostics = build_hotpot_gold_diagnostics(
            sample.gold_titles,
            raw_candidates=raw_candidate_docs,
            deduped_candidates=deduped_docs,
            final_docs=docs,
        )
        unique_title_counts.append(retrieval_diagnostics.unique_titles_in_final_docs)
        raw_candidate_counts.append(retrieval_diagnostics.raw_candidate_count)
        dedup_candidate_counts.append(retrieval_diagnostics.dedup_candidate_count)
        duplicate_candidates_removed_counts.append(retrieval_diagnostics.duplicate_candidates_removed)
        rerank_latencies_ms.append(rerank_latency_ms)

        start_generation = time.perf_counter()
        generation_error = ""
        try:
            generation_result = generator.generate(sample.question, docs)
        except Exception as exc:
            if not continue_on_generation_error:
                raise
            logger.warning("Generation error for %s: %s", sample.query_id, exc)
            generation_errors += 1
            generation_error = f"{type(exc).__name__}: {exc}"
            generation_result = None
        generation_latency_ms = (time.perf_counter() - start_generation) * 1000
        generation_latencies_ms.append(generation_latency_ms)
        predicted = generation_result.text if generation_result is not None else ""

        # --- Phase 5A: answer post-processing (opt-in) ---
        # Strip hedging prefixes before EM/F1 computation so metric normalisation
        # operates on the clean extracted answer rather than the LLM preamble.
        hedging_detected = False
        if postprocess_answers and predicted:
            from src.generation.postprocess import postprocess_answer
            predicted, hedging_detected = postprocess_answer(predicted)

        # --- Phase 5C: citation precision readout ---
        # CitationConstrainedGenerator stores CitationScoringResult in
        # last_citation_result after each generate() call.  We read it via
        # hasattr so this path is a no-op for all other generator types.
        citation_count: int | None = None
        citation_precision: float | None = None
        if (
            generation_result is not None
            and hasattr(generator, "last_citation_result")
            and generator.last_citation_result is not None  # type: ignore[union-attr]
        ):
            cr = generator.last_citation_result  # type: ignore[union-attr]
            citation_count = cr.citation_count
            citation_precision = cr.citation_precision

        approx_input_tokens = _estimate_tokens(sample.question) + sum(_estimate_tokens(d.text) for d in docs)
        approx_output_tokens = _estimate_tokens(predicted)
        input_token_counts.append(approx_input_tokens)
        output_token_counts.append(approx_output_tokens)
        if generation_result is not None and generation_result.input_tokens is not None:
            actual_input_token_counts.append(generation_result.input_tokens)
        if generation_result is not None and generation_result.output_tokens is not None:
            actual_output_token_counts.append(generation_result.output_tokens)
        if generation_result is not None and generation_result.reasoning_tokens is not None:
            actual_reasoning_token_counts.append(generation_result.reasoning_tokens)
        if generation_result is not None and generation_result.cost_usd is not None:
            generation_costs_usd.append(generation_result.cost_usd)

        is_em = exact_match_any(predicted, sample.answers)
        f1 = max_f1_score(predicted, sample.answers)
        r_at_k = _recall_for_sample(sample, docs, top_k)
        results.append(
            RunExampleResult(
                query_id=sample.query_id,
                question=sample.question,
                predicted_answer=predicted,
                gold_answers=sample.answers,
                retrieved_doc_ids=[d.doc_id for d in docs],
                retrieved_titles=[d.title for d in docs],
                retrieved_texts=[d.text for d in docs],
                unique_retrieved_titles=retrieval_diagnostics.unique_titles_in_final_docs,
                retrieval_latency_ms=retrieval_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                generation_latency_ms=generation_latency_ms,
                approx_input_tokens=approx_input_tokens,
                approx_output_tokens=approx_output_tokens,
                actual_input_tokens=generation_result.input_tokens if generation_result is not None else None,
                actual_output_tokens=generation_result.output_tokens if generation_result is not None else None,
                actual_reasoning_tokens=generation_result.reasoning_tokens if generation_result is not None else None,
                generation_cost_usd=generation_result.cost_usd if generation_result is not None else None,
                generation_provider=generation_result.provider if generation_result is not None else "",
                generation_model=generation_result.model if generation_result is not None else "",
                generation_error=generation_error,
                is_em=is_em,
                f1=f1,
                recall_at_k=r_at_k,
                raw_candidate_count=retrieval_diagnostics.raw_candidate_count,
                dedup_candidate_count=retrieval_diagnostics.dedup_candidate_count,
                duplicate_candidates_removed=retrieval_diagnostics.duplicate_candidates_removed,
                dedup_mode=dedup_mode,
                expanded_query=expanded_query,
                expanded_queries=expanded_queries,
                query_expansion_mode=query_expansion_mode,
                query_expansion_latency_ms=query_expansion_latency_ms,
                query_expansion_error=query_expansion_error,
                query_expansion_failure_reason=query_expansion_failure_reason,
                query_expansion_cache_key=query_expansion_cache_key,
                query_expansion_used_fallback=query_expansion_used_fallback,
                gold_titles=list(sample.gold_titles),
                gold_title_ranks=hotpot_gold_diagnostics.gold_title_ranks,
                gold_titles_in_raw_candidates=hotpot_gold_diagnostics.gold_titles_in_raw_candidates,
                gold_titles_after_dedup=hotpot_gold_diagnostics.gold_titles_after_dedup,
                gold_titles_in_final_top_k=hotpot_gold_diagnostics.gold_titles_in_final_top_k,
                missing_gold_count=hotpot_gold_diagnostics.missing_gold_count,
                first_gold_found=hotpot_gold_diagnostics.first_gold_found,
                second_gold_found=hotpot_gold_diagnostics.second_gold_found,
                retrieval_failure_bucket=hotpot_gold_diagnostics.retrieval_failure_bucket,
                # Phase 5 fields
                citation_count=citation_count,
                citation_precision=citation_precision,
                hedging_detected=hedging_detected,
            )
        )
        if progress_callback is not None:
            progress_callback(results, len(results), total_samples)

    metrics = {
        "EM": mean(1.0 if r.is_em else 0.0 for r in results),
        "F1": mean(r.f1 for r in results),
        "Recall@k": mean(r.recall_at_k for r in results),
        "AvgRetrievalLatencyMs": mean(retrieval_latencies_ms) if retrieval_latencies_ms else 0.0,
        "AvgRerankLatencyMs": mean(rerank_latencies_ms) if rerank_latencies_ms else 0.0,
        "AvgGenerationLatencyMs": mean(generation_latencies_ms) if generation_latencies_ms else 0.0,
        "AvgInputTokensApprox": mean(input_token_counts) if input_token_counts else 0.0,
        "AvgOutputTokensApprox": mean(output_token_counts) if output_token_counts else 0.0,
        "AvgInputTokensActual": mean(actual_input_token_counts) if actual_input_token_counts else 0.0,
        "AvgOutputTokensActual": mean(actual_output_token_counts) if actual_output_token_counts else 0.0,
        "AvgReasoningTokensActual": mean(actual_reasoning_token_counts) if actual_reasoning_token_counts else 0.0,
        "AvgUniqueTitles@k": mean(unique_title_counts) if unique_title_counts else 0.0,
        "AvgRawCandidateCount": mean(raw_candidate_counts) if raw_candidate_counts else 0.0,
        "AvgCandidatesAfterDedup": mean(dedup_candidate_counts) if dedup_candidate_counts else 0.0,
        "AvgDuplicateCandidatesRemoved": (
            mean(duplicate_candidates_removed_counts) if duplicate_candidates_removed_counts else 0.0
        ),
        "AvgExpandedQueriesPerSample": mean(expanded_query_counts) if expanded_query_counts else 0.0,
        "AvgQueryExpansionLatencyMs": (
            mean(query_expansion_latencies_ms) if query_expansion_latencies_ms else 0.0
        ),
        "TotalGenerationCostUsd": sum(generation_costs_usd),
        "AvgGenerationCostUsd": mean(generation_costs_usd) if generation_costs_usd else 0.0,
        "AvgLatencyMs": (
            (
                mean(query_expansion_latencies_ms)
                + mean(retrieval_latencies_ms)
                + mean(rerank_latencies_ms)
                + mean(generation_latencies_ms)
            )
            if query_expansion_latencies_ms
            and retrieval_latencies_ms
            and rerank_latencies_ms
            and generation_latencies_ms
            else 0.0
        ),
        "DedupMode": dedup_mode,
        "DedupBeforeRerank": bool(dedup_before_rerank),
        "TitleFirstRerank": bool(title_first_rerank),
        "TitlePoolK": int(title_pool_k),
        "MaxChunksPerTitle": int(max_chunks_per_title),
        "MinUniqueTitles": int(min_unique_titles),
        "QueryExpansionMode": query_expansion_mode,
        "NumQueryExpansionErrors": query_expansion_errors,
        "QueryExpansionErrorRate": (query_expansion_errors / len(results)) if results else 0.0,
        "NumQueryExpansionFallbacks": sum(
            1 for result in results if result.query_expansion_used_fallback
        ),
        "QueryExpansionFallbackRate": (
            sum(1 for result in results if result.query_expansion_used_fallback) / len(results)
            if results
            else 0.0
        ),
        "ContinueOnGenerationError": bool(continue_on_generation_error),
        "NumGenerationErrors": generation_errors,
        "GenerationErrorRate": (generation_errors / len(results)) if results else 0.0,
        # Phase 5 aggregate metrics
        "HedgingRate": (
            sum(1 for r in results if r.hedging_detected) / len(results) if results else 0.0
        ),
        "AvgCitationCount": (
            mean(r.citation_count for r in results if r.citation_count is not None)
            if any(r.citation_count is not None for r in results)
            else 0.0
        ),
        "AvgCitationPrecision": (
            mean(r.citation_precision for r in results if r.citation_precision is not None)
            if any(r.citation_precision is not None for r in results)
            else 0.0
        ),
    }
    metrics.update(_hotpot_metrics(results))
    return results, metrics
