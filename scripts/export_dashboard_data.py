"""Export a self-contained data bundle for the Streamlit dashboard.

Reads aggregated results, failure analysis, and sample predictions,
then writes a single JSON file that the dashboard loads at startup.

Usage:
    uv run python scripts/export_dashboard_data.py \
        --matrix-dir experiments/runs/phase4_matrix/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json_safe(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_sample_predictions(
    matrix_dir: Path,
    max_per_combo: int = 20,
) -> dict[str, list[dict]]:
    """Collect a sample of predictions per config/dataset combo."""
    samples: dict[str, list[dict]] = {}
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
            continue

        with open(pred_path, encoding="utf-8") as f:
            predictions = json.load(f)
        if not isinstance(predictions, list):
            continue

        key = f"{config_name}|{dataset}"
        # Keep only essential fields per record to reduce bundle size
        slim = []
        for record in predictions[:max_per_combo]:
            slim.append({
                "query_id": record.get("query_id", ""),
                "question": record.get("question", record.get("query", "")),
                "gold_answers": record.get("gold_answers", []),
                "predicted_answer": record.get("predicted_answer", ""),
                "f1": record.get("f1", 0),
                "is_em": record.get("is_em", False),
                "recall_at_k": record.get("recall_at_k", 0),
                "retrieved_titles": record.get("retrieved_titles", [])[:10],
                "retrieval_failure_bucket": record.get("retrieval_failure_bucket", ""),
            })
        samples[key] = slim
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dashboard data bundle.")
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=ROOT / "experiments" / "phase4_results.json",
    )
    parser.add_argument(
        "--failure-json",
        type=Path,
        default=ROOT / "experiments" / "phase4_failure_analysis.json",
    )
    parser.add_argument(
        "--case-studies",
        type=Path,
        default=ROOT / "report" / "case_studies.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "app" / "data" / "dashboard_bundle.json",
    )
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    bundle: dict = {}

    # Aggregated results
    results = load_json_safe(args.results_json)
    bundle["results"] = results or []
    print(f"Results: {len(bundle['results'])} rows")

    # Failure analysis
    failures = load_json_safe(args.failure_json)
    bundle["failure_analysis"] = failures or {}
    print(f"Failure analysis: {'loaded' if failures else 'not found'}")

    # Case studies
    cases = load_json_safe(args.case_studies)
    bundle["case_studies"] = cases or []
    print(f"Case studies: {len(bundle['case_studies'])} cases")

    # Sample predictions for case viewer
    samples = collect_sample_predictions(args.matrix_dir, max_per_combo=args.max_samples)
    bundle["sample_predictions"] = samples
    print(f"Sample predictions: {len(samples)} combos")

    # Write bundle
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"\nBundle -> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
