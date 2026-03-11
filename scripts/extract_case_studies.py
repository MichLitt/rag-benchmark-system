"""Extract representative case studies from the Phase 4 experiment matrix.

Finds interesting queries where configs disagree and outputs structured
case study data for the report and dashboard.

Usage:
    uv run python scripts/extract_case_studies.py \
        --matrix-dir experiments/runs/phase4_matrix/ \
        --output report/case_studies.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_predictions(matrix_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """Load all predictions keyed by (config, dataset)."""
    result: dict[tuple[str, str], list[dict]] = {}
    for pred_path in sorted(matrix_dir.rglob("predictions.json")):
        rel = pred_path.relative_to(matrix_dir)
        parts = rel.parts
        config_name = parts[0] if parts else "unknown"
        dataset = ""
        for part in parts:
            if part in ("hotpotqa", "nq", "triviaqa"):
                dataset = part
                break
        if not dataset:
            metrics_path = pred_path.parent / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as f:
                    dataset = json.load(f).get("Dataset", "unknown")
            else:
                dataset = "unknown"
        with open(pred_path, encoding="utf-8") as f:
            predictions = json.load(f)
        if isinstance(predictions, list):
            result[(config_name, dataset)] = predictions
    return result


def build_query_index(
    all_preds: dict[tuple[str, str], list[dict]],
) -> dict[str, dict[str, dict[str, dict]]]:
    """Build index: dataset -> query_id -> config -> record."""
    index: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for (config, dataset), predictions in all_preds.items():
        for record in predictions:
            qid = str(record.get("query_id", ""))
            index[dataset][qid][config] = record
    return dict(index)


def extract_cases(
    index: dict[str, dict[str, dict[str, dict]]],
    max_per_category: int = 3,
) -> list[dict]:
    """Extract representative case studies."""
    cases = []
    categories_found: dict[str, int] = defaultdict(int)

    for dataset, queries in sorted(index.items()):
        for qid, config_records in queries.items():
            configs = list(config_records.keys())
            if len(configs) < 2:
                continue

            f1_by_config = {c: float(r.get("f1", 0)) for c, r in config_records.items()}
            correct_configs = [c for c, f1 in f1_by_config.items() if f1 >= 0.3]
            wrong_configs = [c for c, f1 in f1_by_config.items() if f1 < 0.3]

            # Determine case category
            category = None
            if not correct_configs and wrong_configs:
                category = "all_failed"
            elif correct_configs and wrong_configs:
                # Check if reranker helped
                if any("C1" in c for c in wrong_configs) and any("C2" in c for c in correct_configs):
                    category = "reranker_helped"
                elif any("C4" in c or "hyde" in c.lower() for c in correct_configs):
                    category = "hyde_helped"
                elif any("C5" in c or "decompose" in c.lower() for c in correct_configs):
                    category = "decompose_helped"
                else:
                    category = "config_flip"
            elif correct_configs and not wrong_configs:
                # Check for generation failure (correct but low EM)
                any_record = next(iter(config_records.values()))
                if not any_record.get("is_em", False) and float(any_record.get("f1", 0)) < 0.6:
                    category = "partial_answer"

            if category is None:
                continue
            if categories_found[category] >= max_per_category:
                continue
            categories_found[category] += 1

            # Build case study record
            first_record = next(iter(config_records.values()))
            case = {
                "category": category,
                "dataset": dataset,
                "query_id": qid,
                "question": first_record.get("question", first_record.get("query", "")),
                "gold_answers": first_record.get("gold_answers", []),
                "configs": {},
            }
            for config, record in config_records.items():
                case["configs"][config] = {
                    "predicted_answer": record.get("predicted_answer", ""),
                    "f1": float(record.get("f1", 0)),
                    "is_em": bool(record.get("is_em", False)),
                    "retrieved_titles": record.get("retrieved_titles", [])[:5],
                    "recall_at_k": float(record.get("recall_at_k", 0)),
                }
            cases.append(case)

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract case studies from experiment matrix.")
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "report" / "case_studies.json",
    )
    parser.add_argument("--max-per-category", type=int, default=3)
    args = parser.parse_args()

    all_preds = discover_predictions(args.matrix_dir)
    if not all_preds:
        print(f"No predictions found under {args.matrix_dir}")
        sys.exit(1)

    print(f"Loaded predictions from {len(all_preds)} (config, dataset) pairs.")

    index = build_query_index(all_preds)
    cases = extract_cases(index, max_per_category=args.max_per_category)

    # Group by category for summary
    by_cat: dict[str, int] = defaultdict(int)
    for case in cases:
        by_cat[case["category"]] += 1

    print(f"\nExtracted {len(cases)} case studies:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
