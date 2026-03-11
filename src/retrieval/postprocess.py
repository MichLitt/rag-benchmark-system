from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from src.types import Document


def normalize_title(title: str) -> str:
    return " ".join(title.lower().strip().split())


@dataclass(frozen=True)
class RetrievalDiagnostics:
    raw_candidate_count: int
    dedup_candidate_count: int
    duplicate_candidates_removed: int
    unique_titles_in_final_docs: int
    final_doc_count: int


@dataclass(frozen=True)
class HotpotGoldDiagnostics:
    gold_title_ranks: dict[str, int | None]
    gold_titles_in_raw_candidates: list[str]
    gold_titles_after_dedup: list[str]
    gold_titles_in_final_top_k: list[str]
    missing_gold_count: int
    first_gold_found: bool
    second_gold_found: bool
    retrieval_failure_bucket: str


def _document_key(doc: Document, mode: str) -> str:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "doc_id":
        key = doc.doc_id.strip()
    else:
        key = normalize_title(doc.title) or doc.doc_id.strip()
    if key:
        return key
    return doc.text.strip()


def deduplicate_documents(docs: list[Document], mode: str) -> list[Document]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"", "off", "none"}:
        return list(docs)
    if normalized_mode not in {"title", "doc_id"}:
        raise ValueError(f"Unsupported dedup mode: {mode}")

    seen: set[str] = set()
    deduped: list[Document] = []
    for doc in docs:
        key = _document_key(doc, normalized_mode)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def select_title_representatives(docs: list[Document], max_titles: int) -> list[Document]:
    if max_titles <= 0:
        return []

    representatives: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        key = _document_key(doc, "title")
        if key in seen:
            continue
        seen.add(key)
        representatives.append(doc)
        if len(representatives) >= max_titles:
            break
    return representatives


def pack_title_diverse_documents(
    ranked_title_docs: list[Document],
    raw_candidate_docs: list[Document],
    top_k: int,
    *,
    max_chunks_per_title: int,
    min_unique_titles: int,
) -> list[Document]:
    if top_k <= 0:
        return []
    if max_chunks_per_title <= 0:
        raise ValueError(f"max_chunks_per_title must be > 0, got {max_chunks_per_title}")
    if min_unique_titles < 0:
        raise ValueError(f"min_unique_titles must be >= 0, got {min_unique_titles}")

    grouped: OrderedDict[str, list[Document]] = OrderedDict()
    for doc in raw_candidate_docs:
        key = _document_key(doc, "title")
        grouped.setdefault(key, []).append(doc)

    ranked_keys: list[str] = []
    seen_ranked: set[str] = set()
    for doc in ranked_title_docs:
        key = _document_key(doc, "title")
        if key in seen_ranked:
            continue
        if key not in grouped:
            continue
        seen_ranked.add(key)
        ranked_keys.append(key)
    for key in grouped:
        if key not in seen_ranked:
            ranked_keys.append(key)

    packed: list[Document] = []
    used_doc_ids: set[str] = set()

    max_unique_targets = min(
        top_k,
        max(len(ranked_keys), min_unique_titles),
    )
    for key in ranked_keys:
        docs_for_title = grouped.get(key, [])
        if not docs_for_title:
            continue
        doc = docs_for_title[0]
        doc_key = _document_key(doc, "doc_id")
        if doc_key in used_doc_ids:
            continue
        packed.append(doc)
        used_doc_ids.add(doc_key)
        if len(packed) >= max_unique_targets:
            break

    if len(packed) >= top_k:
        return packed[:top_k]

    for chunk_index in range(1, max_chunks_per_title):
        for key in ranked_keys:
            docs_for_title = grouped.get(key, [])
            if chunk_index >= len(docs_for_title):
                continue
            doc = docs_for_title[chunk_index]
            doc_key = _document_key(doc, "doc_id")
            if doc_key in used_doc_ids:
                continue
            packed.append(doc)
            used_doc_ids.add(doc_key)
            if len(packed) >= top_k:
                return packed[:top_k]

    return packed[:top_k]


def unique_title_count(docs: list[Document]) -> int:
    titles = {normalize_title(doc.title) for doc in docs if doc.title.strip()}
    if titles:
        return len(titles)
    doc_ids = {doc.doc_id.strip() for doc in docs if doc.doc_id.strip()}
    return len(doc_ids)


def build_hotpot_gold_diagnostics(
    gold_titles: list[str],
    raw_candidates: list[Document],
    deduped_candidates: list[Document],
    final_docs: list[Document],
) -> HotpotGoldDiagnostics:
    ordered_gold_titles: list[str] = []
    seen_gold: set[str] = set()
    for title in gold_titles:
        normalized = normalize_title(title)
        if not normalized or normalized in seen_gold:
            continue
        seen_gold.add(normalized)
        ordered_gold_titles.append(title)

    if not ordered_gold_titles:
        return HotpotGoldDiagnostics(
            gold_title_ranks={},
            gold_titles_in_raw_candidates=[],
            gold_titles_after_dedup=[],
            gold_titles_in_final_top_k=[],
            missing_gold_count=0,
            first_gold_found=False,
            second_gold_found=False,
            retrieval_failure_bucket="not_applicable",
        )

    normalized_gold = [normalize_title(title) for title in ordered_gold_titles]
    raw_titles = {normalize_title(doc.title) for doc in raw_candidates if doc.title.strip()}
    dedup_titles = {normalize_title(doc.title) for doc in deduped_candidates if doc.title.strip()}
    final_title_ranks: dict[str, int] = {}
    for rank, doc in enumerate(final_docs, start=1):
        normalized = normalize_title(doc.title)
        if normalized and normalized not in final_title_ranks:
            final_title_ranks[normalized] = rank

    raw_hits = [
        title for title, normalized in zip(ordered_gold_titles, normalized_gold)
        if normalized in raw_titles
    ]
    dedup_hits = [
        title for title, normalized in zip(ordered_gold_titles, normalized_gold)
        if normalized in dedup_titles
    ]
    final_hits = [
        title for title, normalized in zip(ordered_gold_titles, normalized_gold)
        if normalized in final_title_ranks
    ]
    gold_title_ranks = {
        title: final_title_ranks.get(normalized)
        for title, normalized in zip(ordered_gold_titles, normalized_gold)
    }

    if len(raw_hits) == 0:
        bucket = "no_gold_in_raw"
    elif len(raw_hits) == 1:
        bucket = "only_one_gold_in_raw"
    elif len(dedup_hits) < len(ordered_gold_titles):
        bucket = "both_gold_in_raw_but_lost_after_dedup"
    elif len(final_hits) < len(ordered_gold_titles):
        bucket = "both_gold_after_dedup_but_lost_after_rerank"
    else:
        bucket = "both_gold_in_final"

    return HotpotGoldDiagnostics(
        gold_title_ranks=gold_title_ranks,
        gold_titles_in_raw_candidates=raw_hits,
        gold_titles_after_dedup=dedup_hits,
        gold_titles_in_final_top_k=final_hits,
        missing_gold_count=max(0, len(ordered_gold_titles) - len(final_hits)),
        first_gold_found=bool(normalized_gold) and normalized_gold[0] in final_title_ranks,
        second_gold_found=len(normalized_gold) >= 2 and normalized_gold[1] in final_title_ranks,
        retrieval_failure_bucket=bucket,
    )


def build_retrieval_diagnostics(
    raw_candidates: list[Document],
    deduped_candidates: list[Document],
    final_docs: list[Document],
) -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        raw_candidate_count=len(raw_candidates),
        dedup_candidate_count=len(deduped_candidates),
        duplicate_candidates_removed=max(0, len(raw_candidates) - len(deduped_candidates)),
        unique_titles_in_final_docs=unique_title_count(final_docs),
        final_doc_count=len(final_docs),
    )
