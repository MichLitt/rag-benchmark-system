"""Generic failure mode classification for datasets without gold doc IDs.

For NQ/TriviaQA where we don't have gold_doc_ids or gold_titles, failures
are classified based on answer-presence heuristics in retrieved text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class GenericFailureMode(str, Enum):
    CORRECT = "correct"
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"


@dataclass
class GenericFailureModeResult:
    query_id: str
    failure_mode: GenericFailureMode
    f1: float
    is_em: bool
    predicted_answer: str
    gold_answers: list[str]
    answer_found_in_context: bool


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _answer_in_retrieved_text(
    gold_answers: list[str],
    retrieved_titles: list[str],
    retrieved_texts: list[str],
) -> bool:
    """Check if any gold answer appears as a substring in retrieved context."""
    normalized_answers = [_normalize(a) for a in gold_answers if a.strip()]
    if not normalized_answers:
        return False

    for title, text in zip(retrieved_titles, retrieved_texts):
        haystack = _normalize(f"{title} {text}")
        for answer in normalized_answers:
            if answer and answer in haystack:
                return True
    return False


def classify_record(
    record: dict,
    f1_threshold: float = 0.3,
) -> GenericFailureModeResult:
    """Classify one predictions.json record into a generic failure mode."""
    query_id = str(record.get("query_id", ""))
    f1 = float(record.get("f1", 0.0))
    is_em = bool(record.get("is_em", False))
    predicted_answer = str(record.get("predicted_answer", ""))
    gold_answers = [str(a) for a in record.get("gold_answers", [])]
    retrieved_titles = [str(t) for t in record.get("retrieved_titles", [])]
    retrieved_texts = [str(t) for t in record.get("retrieved_texts", [])]

    if f1 >= f1_threshold:
        mode = GenericFailureMode.CORRECT
        answer_found = True
    else:
        answer_found = _answer_in_retrieved_text(
            gold_answers, retrieved_titles, retrieved_texts,
        )
        if answer_found:
            mode = GenericFailureMode.GENERATION_FAILURE
        else:
            mode = GenericFailureMode.RETRIEVAL_FAILURE

    return GenericFailureModeResult(
        query_id=query_id,
        failure_mode=mode,
        f1=f1,
        is_em=is_em,
        predicted_answer=predicted_answer,
        gold_answers=gold_answers,
        answer_found_in_context=answer_found,
    )


def classify_all(
    predictions: list[dict],
    f1_threshold: float = 0.3,
) -> list[GenericFailureModeResult]:
    return [classify_record(record, f1_threshold) for record in predictions]


def summarize(results: list[GenericFailureModeResult]) -> dict:
    total = len(results)
    counts: dict[str, int] = {mode.value: 0 for mode in GenericFailureMode}
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
