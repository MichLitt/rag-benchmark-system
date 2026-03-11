"""Failure mode classification utilities for Hotpot-style retrieval debugging."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    CORRECT = "correct"
    NO_GOLD_IN_RAW = "no_gold_in_raw"
    ONLY_ONE_GOLD_IN_RAW = "only_one_gold_in_raw"
    LOST_AFTER_DEDUP = "both_gold_in_raw_but_lost_after_dedup"
    LOST_AFTER_RERANK = "both_gold_after_dedup_but_lost_after_rerank"
    BOTH_GOLD_IN_FINAL = "both_gold_in_final"
    GENERATION = "generation_failure"


@dataclass
class FailureModeResult:
    query_id: str
    failure_mode: FailureMode
    f1: float
    is_em: bool
    recall_at_k: float
    gold_titles: list[str]
    retrieved_titles: list[str]
    predicted_answer: str
    gold_answers: list[str]


def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"[_-]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title


def _gold_in_corpus(gold_titles: list[str], corpus_titles: set[str]) -> bool:
    for title in gold_titles:
        if _normalize_title(title) in corpus_titles:
            return True
    return False


def _gold_in_top_n(gold_titles: list[str], retrieved_titles: list[str], n: int) -> bool:
    top_n_norm = {_normalize_title(title) for title in retrieved_titles[:n]}
    for title in gold_titles:
        if _normalize_title(title) in top_n_norm:
            return True
    return False


def classify_record(
    record: dict,
    corpus_titles: set[str] | None = None,
    f1_threshold: float = 0.3,
    good_rank_cutoff: int = 2,
) -> FailureModeResult:
    """Classify one predictions.json record into a failure mode."""
    query_id = str(record.get("query_id", ""))
    f1 = float(record.get("f1", 0.0))
    is_em = bool(record.get("is_em", False))
    recall_at_k = float(record.get("recall_at_k", 0.0))
    gold_titles = [str(title) for title in record.get("gold_titles", [])]
    retrieved_titles = [str(title) for title in record.get("retrieved_titles", [])]
    predicted_answer = str(record.get("predicted_answer", ""))
    gold_answers = [str(answer) for answer in record.get("gold_answers", [])]
    retrieval_failure_bucket = str(record.get("retrieval_failure_bucket", "")).strip()

    if f1 >= f1_threshold:
        mode = FailureMode.CORRECT
    elif retrieval_failure_bucket == FailureMode.NO_GOLD_IN_RAW.value:
        mode = FailureMode.NO_GOLD_IN_RAW
    elif retrieval_failure_bucket == FailureMode.ONLY_ONE_GOLD_IN_RAW.value:
        mode = FailureMode.ONLY_ONE_GOLD_IN_RAW
    elif retrieval_failure_bucket == FailureMode.LOST_AFTER_DEDUP.value:
        mode = FailureMode.LOST_AFTER_DEDUP
    elif retrieval_failure_bucket == FailureMode.LOST_AFTER_RERANK.value:
        mode = FailureMode.LOST_AFTER_RERANK
    elif retrieval_failure_bucket == FailureMode.BOTH_GOLD_IN_FINAL.value:
        mode = FailureMode.BOTH_GOLD_IN_FINAL
    elif corpus_titles is not None and gold_titles and not _gold_in_corpus(gold_titles, corpus_titles):
        mode = FailureMode.NO_GOLD_IN_RAW
    elif recall_at_k == 0.0:
        mode = FailureMode.NO_GOLD_IN_RAW
    elif not _gold_in_top_n(gold_titles, retrieved_titles, good_rank_cutoff):
        mode = FailureMode.LOST_AFTER_RERANK
    else:
        mode = FailureMode.GENERATION

    return FailureModeResult(
        query_id=query_id,
        failure_mode=mode,
        f1=f1,
        is_em=is_em,
        recall_at_k=recall_at_k,
        gold_titles=gold_titles,
        retrieved_titles=retrieved_titles,
        predicted_answer=predicted_answer,
        gold_answers=gold_answers,
    )


def classify_all(
    predictions: list[dict],
    corpus_titles: set[str] | None = None,
    f1_threshold: float = 0.3,
    good_rank_cutoff: int = 2,
) -> list[FailureModeResult]:
    return [
        classify_record(record, corpus_titles, f1_threshold, good_rank_cutoff)
        for record in predictions
    ]


def summarize(results: list[FailureModeResult]) -> dict:
    total = len(results)
    counts: dict[str, int] = {mode.value: 0 for mode in FailureMode}
    for result in results:
        counts[result.failure_mode.value] += 1

    def _pct(count: int) -> float:
        return round(count / total * 100, 2) if total else 0.0

    return {
        "total": total,
        **{
            mode: {"count": counts[mode], "pct": _pct(counts[mode])}
            for mode in counts
        },
    }
