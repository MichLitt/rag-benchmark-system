from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_TAXONOMY_EXAMPLES_PATH = Path(
    "experiments/runs/stage2_hotpot_matrix/"
    "hotpot_retrieval_dense_sharded_20260308_132925/"
    "failure_taxonomy/taxonomy_examples.jsonl"
)
DEFAULT_TAXONOMY_SUMMARY_PATH = Path(
    "experiments/runs/stage2_hotpot_matrix/"
    "hotpot_retrieval_dense_sharded_20260308_132925/"
    "failure_taxonomy/taxonomy_summary.json"
)
DEFAULT_QA_PATH = Path("data/raw/flashrag/hotpotqa/dev/qa.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/filtered/hotpotqa_query_screening_50.jsonl")
DEFAULT_TARGET_BUCKETS = OrderedDict(
    [
        ("query_formulation_gap", 25),
        ("normalization_or_alias_suspect", 15),
        ("budget_limited", 10),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reproducible 50-query Hotpot screening subset.")
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--taxonomy-examples-path", type=Path, default=DEFAULT_TAXONOMY_EXAMPLES_PATH)
    parser.add_argument("--taxonomy-summary-path", type=Path, default=DEFAULT_TAXONOMY_SUMMARY_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_blocker_order(summary_path: Path | None) -> list[str]:
    if summary_path is None or not summary_path.exists():
        return list(DEFAULT_TARGET_BUCKETS.keys())
    payload = _load_json(summary_path)
    if not isinstance(payload, dict):
        return list(DEFAULT_TARGET_BUCKETS.keys())
    blockers = payload.get("top_blockers", [])
    if not isinstance(blockers, list):
        return list(DEFAULT_TARGET_BUCKETS.keys())
    ordered = [
        str(item.get("subcategory", "")).strip()
        for item in blockers
        if isinstance(item, dict) and str(item.get("subcategory", "")).strip()
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for bucket in ordered + list(DEFAULT_TARGET_BUCKETS.keys()):
        if bucket and bucket not in seen:
            deduped.append(bucket)
            seen.add(bucket)
    return deduped


def _select_ids_in_source_order(
    qa_rows: list[dict],
    taxonomy_by_id: dict[str, str],
    bucket: str,
    selected_ids: set[str],
    limit: int,
) -> list[str]:
    chosen: list[str] = []
    for row in qa_rows:
        query_id = str(row.get("id", "")).strip()
        if not query_id or query_id in selected_ids:
            continue
        if taxonomy_by_id.get(query_id) != bucket:
            continue
        chosen.append(query_id)
        if len(chosen) >= limit:
            break
    return chosen


def main() -> None:
    args = parse_args()
    qa_rows = _load_jsonl(args.qa_path)
    taxonomy_rows = _load_jsonl(args.taxonomy_examples_path)
    taxonomy_by_id = {
        str(row.get("query_id", "")).strip(): str(row.get("subcategory", "")).strip()
        for row in taxonomy_rows
        if str(row.get("query_id", "")).strip()
    }
    blocker_order = _load_blocker_order(args.taxonomy_summary_path)

    selected_ids: set[str] = set()
    selected_by_bucket: dict[str, list[str]] = {bucket: [] for bucket in DEFAULT_TARGET_BUCKETS}
    remaining_slots_by_bucket: dict[str, int] = {}

    for bucket, target_count in DEFAULT_TARGET_BUCKETS.items():
        chosen = _select_ids_in_source_order(
            qa_rows=qa_rows,
            taxonomy_by_id=taxonomy_by_id,
            bucket=bucket,
            selected_ids=selected_ids,
            limit=target_count,
        )
        selected_ids.update(chosen)
        selected_by_bucket[bucket] = chosen
        remaining_slots_by_bucket[bucket] = max(0, target_count - len(chosen))

    for bucket, remaining in remaining_slots_by_bucket.items():
        if remaining <= 0:
            continue
        for fallback_bucket in blocker_order:
            if fallback_bucket == bucket:
                continue
            refill = _select_ids_in_source_order(
                qa_rows=qa_rows,
                taxonomy_by_id=taxonomy_by_id,
                bucket=fallback_bucket,
                selected_ids=selected_ids,
                limit=remaining,
            )
            selected_ids.update(refill)
            selected_by_bucket[bucket].extend(refill)
            remaining -= len(refill)
            if remaining <= 0:
                break
        remaining_slots_by_bucket[bucket] = remaining

    unresolved = {bucket: remaining for bucket, remaining in remaining_slots_by_bucket.items() if remaining > 0}
    if unresolved:
        raise RuntimeError(f"Unable to fill screening subset targets: {unresolved}")

    selected_rows = [
        row for row in qa_rows if str(row.get("id", "")).strip() in selected_ids
    ]
    expected_total = sum(DEFAULT_TARGET_BUCKETS.values())
    if len(selected_rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} selected rows, got {len(selected_rows)}."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    actual_bucket_counts: dict[str, int] = {}
    for row in selected_rows:
        bucket = taxonomy_by_id.get(str(row.get("id", "")).strip(), "unlabeled")
        actual_bucket_counts[bucket] = actual_bucket_counts.get(bucket, 0) + 1

    print(
        json.dumps(
            {
                "output_path": str(args.output_path.resolve()),
                "selected_queries": len(selected_rows),
                "target_bucket_counts": dict(DEFAULT_TARGET_BUCKETS),
                "actual_bucket_counts": actual_bucket_counts,
                "blocker_order": blocker_order,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
