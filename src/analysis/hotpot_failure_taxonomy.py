from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.retrieval.postprocess import normalize_title
from src.retrieval.tokenize import simple_tokenize


MAIN_CLASS_ORDER = [
    "no_gold_in_raw",
    "only_one_gold_in_raw",
    "both_gold_after_dedup_but_lost_after_rerank",
    "both_gold_in_final",
]

SUBCATEGORY_ORDER = [
    "budget_limited",
    "embedding_confusion",
    "normalization_or_alias_suspect",
    "query_formulation_gap",
    "rerank_loss",
    "resolved",
]

SUBCATEGORY_RECOMMENDATIONS = {
    "budget_limited": "increase raw dense depth or add title-aware prefilter before truncating candidates",
    "embedding_confusion": "add lexical title prior or hybrid title retrieval before reranking",
    "normalization_or_alias_suspect": "tighten title normalization and alias handling before candidate evaluation",
    "query_formulation_gap": "prioritize query rewrite or hotpot_decompose instead of retriever-only tuning",
    "rerank_loss": "adjust reranker or title packing because both gold titles already survived raw retrieval",
    "resolved": "keep as a control group and do not optimize specifically for these samples",
}


@dataclass(frozen=True)
class TaxonomyRecord:
    query_id: str
    question: str
    gold_titles: list[str]
    retrieved_titles: list[str]
    main_class: str
    subcategory: str
    recommendation: str
    missing_gold_titles: list[str]
    dense_probe_hit_titles: list[str]
    sparse_probe_hit_titles: list[str]
    alias_candidates: dict[str, list[str]]
    retrieval_failure_bucket: str
    raw_candidate_count: int
    dedup_candidate_count: int
    title_pool_count: int


def canonical_main_class(bucket: str) -> str:
    normalized = str(bucket or "").strip()
    if normalized == "both_gold_in_raw_but_lost_after_dedup":
        return "both_gold_after_dedup_but_lost_after_rerank"
    if normalized in MAIN_CLASS_ORDER:
        return normalized
    return "only_one_gold_in_raw"


def _strip_parenthetical(title: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", title).strip()


def _approximate_title_match(gold_title: str, candidate_title: str) -> bool:
    gold_norm = normalize_title(gold_title)
    candidate_norm = normalize_title(candidate_title)
    if not gold_norm or not candidate_norm:
        return False
    if gold_norm == candidate_norm:
        return True

    stripped_gold = normalize_title(_strip_parenthetical(gold_title))
    stripped_candidate = normalize_title(_strip_parenthetical(candidate_title))
    if stripped_gold and stripped_gold == stripped_candidate:
        return True

    gold_tokens = set(simple_tokenize(stripped_gold))
    candidate_tokens = set(simple_tokenize(stripped_candidate))
    if not gold_tokens or not candidate_tokens:
        return False
    overlap = len(gold_tokens & candidate_tokens)
    min_size = min(len(gold_tokens), len(candidate_tokens))
    return min_size > 0 and (overlap / min_size) >= 0.8


def find_alias_candidates(
    missing_gold_titles: list[str],
    candidate_titles: Iterable[str],
) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    candidate_list = [str(title) for title in candidate_titles if str(title).strip()]
    for gold_title in missing_gold_titles:
        matched = [title for title in candidate_list if _approximate_title_match(gold_title, title)]
        if matched:
            matches[gold_title] = matched[:5]
    return matches


def load_json(path: str | Path) -> dict | list:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_query_id(record: dict) -> str:
    query_id = str(record.get("id", "")).strip()
    if query_id:
        return query_id
    return str(record.get("query_id", "")).strip()


def write_json(path: str | Path, payload: dict | list) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify_taxonomy_record(
    record: dict,
    *,
    dense_probe_titles: list[str] | None = None,
    sparse_probe_titles: list[str] | None = None,
) -> TaxonomyRecord:
    main_class = canonical_main_class(str(record.get("retrieval_failure_bucket", "")))
    gold_titles = [str(title) for title in record.get("gold_titles", []) if str(title).strip()]
    raw_hits = {
        normalize_title(title)
        for title in record.get("gold_titles_in_raw_candidates", [])
        if str(title).strip()
    }
    missing_gold_titles = [
        title for title in gold_titles
        if normalize_title(title) and normalize_title(title) not in raw_hits
    ]

    dense_probe_titles = [str(title) for title in (dense_probe_titles or []) if str(title).strip()]
    sparse_probe_titles = [str(title) for title in (sparse_probe_titles or []) if str(title).strip()]

    dense_probe_hits = [
        title for title in missing_gold_titles
        if normalize_title(title) in {normalize_title(candidate) for candidate in dense_probe_titles}
    ]
    sparse_probe_hits = [
        title for title in missing_gold_titles
        if normalize_title(title) in {normalize_title(candidate) for candidate in sparse_probe_titles}
    ]
    alias_candidates = find_alias_candidates(
        missing_gold_titles,
        list(record.get("retrieved_titles", [])) + dense_probe_titles + sparse_probe_titles,
    )

    if main_class == "both_gold_in_final":
        subcategory = "resolved"
    elif main_class == "both_gold_after_dedup_but_lost_after_rerank":
        subcategory = "rerank_loss"
    elif dense_probe_hits:
        subcategory = "budget_limited"
    elif sparse_probe_hits:
        subcategory = "embedding_confusion"
    elif alias_candidates:
        subcategory = "normalization_or_alias_suspect"
    else:
        subcategory = "query_formulation_gap"

    return TaxonomyRecord(
        query_id=resolve_query_id(record),
        question=str(record.get("question", "")),
        gold_titles=gold_titles,
        retrieved_titles=[str(title) for title in record.get("retrieved_titles", []) if str(title).strip()],
        main_class=main_class,
        subcategory=subcategory,
        recommendation=SUBCATEGORY_RECOMMENDATIONS[subcategory],
        missing_gold_titles=missing_gold_titles,
        dense_probe_hit_titles=dense_probe_hits,
        sparse_probe_hit_titles=sparse_probe_hits,
        alias_candidates=alias_candidates,
        retrieval_failure_bucket=str(record.get("retrieval_failure_bucket", "")),
        raw_candidate_count=int(record.get("raw_candidate_count", 0) or 0),
        dedup_candidate_count=int(record.get("dedup_candidate_count", 0) or 0),
        title_pool_count=int(record.get("title_pool_count", 0) or 0),
    )


def summarize_taxonomy(records: list[TaxonomyRecord], metrics: dict | None = None) -> dict:
    total = len(records)
    main_counter = Counter(record.main_class for record in records)
    subtype_counter = Counter(record.subcategory for record in records)
    blocker_counter = Counter(
        record.subcategory for record in records if record.subcategory not in {"resolved"}
    )

    summary = {
        "total_examples": total,
        "main_class_counts": {
            key: {
                "count": int(main_counter.get(key, 0)),
                "pct": (main_counter.get(key, 0) / total if total else 0.0),
            }
            for key in MAIN_CLASS_ORDER
        },
        "subcategory_counts": {
            key: {
                "count": int(subtype_counter.get(key, 0)),
                "pct": (subtype_counter.get(key, 0) / total if total else 0.0),
                "recommendation": SUBCATEGORY_RECOMMENDATIONS[key],
            }
            for key in SUBCATEGORY_ORDER
        },
        "top_blockers": [
            {
                "subcategory": subcategory,
                "count": int(count),
                "recommendation": SUBCATEGORY_RECOMMENDATIONS[subcategory],
            }
            for subcategory, count in blocker_counter.most_common()
        ],
        "metrics_snapshot": metrics or {},
    }
    return summary


def build_markdown_report(
    *,
    summary: dict,
    records: list[TaxonomyRecord],
    details_path: Path,
    metrics_path: Path | None,
    manifest_path: Path | None,
    title_bm25_manifest_path: Path | None,
    probe_top_k: int,
    title_probe_top_k: int,
) -> str:
    subtype_groups: dict[str, list[TaxonomyRecord]] = {key: [] for key in SUBCATEGORY_ORDER}
    for record in records:
        subtype_groups.setdefault(record.subcategory, []).append(record)

    lines = [
        "# Hotpot Failure Taxonomy Report",
        "",
        "## 1. Inputs",
        "",
        f"- details: `{details_path}`",
        f"- metrics: `{metrics_path}`" if metrics_path is not None else "- metrics: `not provided`",
        f"- dense manifest: `{manifest_path}`" if manifest_path is not None else "- dense manifest: `not provided`",
        (
            f"- title BM25 manifest: `{title_bm25_manifest_path}`"
            if title_bm25_manifest_path is not None
            else "- title BM25 manifest: `not provided`"
        ),
        f"- dense probe top-k: `{probe_top_k}`",
        f"- title probe top-k: `{title_probe_top_k}`",
        "",
        "## 2. Main Class Counts",
        "",
        "| Main Class | Count | Pct |",
        "| --- | ---: | ---: |",
    ]
    for key in MAIN_CLASS_ORDER:
        item = summary["main_class_counts"][key]
        lines.append(f"| `{key}` | `{item['count']}` | `{item['pct']:.4f}` |")

    lines.extend(
        [
            "",
            "## 3. Subcategory Counts",
            "",
            "| Subcategory | Count | Pct | Recommendation |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for key in SUBCATEGORY_ORDER:
        item = summary["subcategory_counts"][key]
        lines.append(
            f"| `{key}` | `{item['count']}` | `{item['pct']:.4f}` | {item['recommendation']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Top Blockers",
            "",
        ]
    )
    if summary["top_blockers"]:
        for index, blocker in enumerate(summary["top_blockers"], start=1):
            lines.append(
                f"{index}. `{blocker['subcategory']}`: {blocker['count']} "
                f"({blocker['recommendation']})"
            )
    else:
        lines.append("- no blockers")

    lines.extend(
        [
            "",
            "## 5. Representative Examples",
            "",
        ]
    )
    for subcategory in SUBCATEGORY_ORDER:
        examples = subtype_groups.get(subcategory, [])[:10]
        if not examples:
            continue
        lines.extend([f"### `{subcategory}`", ""])
        for example in examples:
            lines.append(f"- `{example.query_id}` {example.question}")
            lines.append(f"  - gold: {example.gold_titles}")
            lines.append(f"  - missing_gold: {example.missing_gold_titles}")
            lines.append(f"  - dense_probe_hits: {example.dense_probe_hit_titles}")
            lines.append(f"  - sparse_probe_hits: {example.sparse_probe_hit_titles}")
            if example.alias_candidates:
                lines.append(f"  - alias_candidates: {example.alias_candidates}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def merge_qa_fields(details: list[dict], qa_rows: list[dict]) -> list[dict]:
    qa_by_id = {str(row.get("id", "")): row for row in qa_rows}
    merged: list[dict] = []
    for detail in details:
        query_id = resolve_query_id(detail)
        qa = qa_by_id.get(query_id, {})
        row = dict(detail)
        if not row.get("id") and query_id:
            row["id"] = query_id
        if not row.get("question") and qa.get("question"):
            row["question"] = qa["question"]
        if not row.get("gold_titles") and qa.get("gold_titles"):
            row["gold_titles"] = qa["gold_titles"]
        merged.append(row)
    return merged


def records_to_dicts(records: list[TaxonomyRecord]) -> list[dict]:
    return [asdict(record) for record in records]
