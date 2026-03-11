from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize HotpotQA retrieval diagnostic details.")
    parser.add_argument(
        "--details-path",
        type=Path,
        required=True,
        help="Path to details.json produced by eval_hotpot_retrieval.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.details_path.read_text(encoding="utf-8"))
    total = len(rows)
    multi_gold = [row for row in rows if len(row.get("gold_titles", [])) >= 2]
    duplicate_topk = [row for row in rows if row.get("duplicate_candidates_removed", 0) > 0]
    second_gold_missing = [row for row in multi_gold if row.get("second_gold_missing")]
    duplicate_and_missing = [
        row for row in second_gold_missing if row.get("duplicate_candidates_removed", 0) > 0
    ]

    report = {
        "details_path": str(args.details_path),
        "queries": total,
        "queries_with_multi_gold_titles": len(multi_gold),
        "queries_with_duplicate_candidates_removed": len(duplicate_topk),
        "queries_with_second_gold_missing": len(second_gold_missing),
        "queries_with_duplicate_loss_and_second_gold_missing": len(duplicate_and_missing),
        "duplicate_candidate_query_pct": (len(duplicate_topk) / total if total else 0.0),
        "second_gold_missing_pct": (len(second_gold_missing) / len(multi_gold) if multi_gold else 0.0),
        "duplicate_loss_and_second_gold_missing_pct": (
            len(duplicate_and_missing) / len(multi_gold) if multi_gold else 0.0
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
