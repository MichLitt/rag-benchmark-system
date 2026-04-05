"""Aggregate experiment results from a Phase 4 matrix into CSV and JSON.

Reads all metrics.json + faithfulness_metrics.json files and produces
a consolidated results table.

Usage:
    uv run python scripts/aggregate_experiment_results.py \
        --matrix-dir experiments/runs/phase4_matrix/

    uv run python scripts/aggregate_experiment_results.py \
        --matrix-dir experiments/runs/phase4_matrix/ \
        --output-csv experiments/phase4_results.csv \
        --output-json experiments/phase4_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METRIC_COLUMNS = [
    "Config",
    "Dataset",
    "NumQueries",
    "EM",
    "F1",
    "Recall@k",
    "AvgFaithfulness",
    "HallucinationRate",
    "AvgLatencyMs",
    "AvgRetrievalLatencyMs",
    "AvgRerankLatencyMs",
    "AvgGenerationLatencyMs",
    "AvgQueryExpansionLatencyMs",
    "TotalGenerationCostUsd",
    "AvgGenerationCostUsd",
    "GenerationErrorRate",
    "UseReranker",
    "QueryExpansionMode",
    "GeneratorModel",
    # A3 NLI citation metrics (empty for datasets that don't compute them)
    "AvgAnswerAttributionRate",
    "AvgSupportingPassageHit",
    "AvgPageGroundingAccuracy",
]


def discover_runs(
    matrix_dir: Path,
    min_queries: int = 50,
) -> list[tuple[str, str, Path]]:
    """Discover (config_name, dataset, metrics_path) tuples.

    Only includes runs with at least min_queries to exclude smoke-test runs.
    When multiple runs exist for the same (config, dataset), picks the largest.
    """
    # best[(config, dataset)] = (num_queries, metrics_path)
    best: dict[tuple[str, str], tuple[int, Path]] = {}

    for config_dir in sorted(matrix_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        for metrics_path in sorted(config_dir.rglob("metrics.json")):
            # Determine dataset from path hierarchy
            rel = metrics_path.relative_to(config_dir)
            parts = rel.parts
            # Typical: run_name/dataset/metrics.json or dataset/metrics.json
            dataset = ""
            for part in parts:
                if part in ("hotpotqa", "nq", "triviaqa"):
                    dataset = part
                    break
            if not dataset:
                # Try to infer from metrics content
                try:
                    with open(metrics_path, encoding="utf-8") as f:
                        data = json.load(f)
                    dataset = data.get("Dataset", "unknown")
                except Exception:
                    dataset = "unknown"

            # Read NumQueries to filter smoke runs and pick best
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    data = json.load(f)
                nq = int(data.get("NumQueries", 0))
            except Exception:
                nq = 0

            if nq < min_queries:
                continue

            key = (config_name, dataset)
            if key not in best or nq > best[key][0]:
                best[key] = (nq, metrics_path)

    return [(config, ds, path) for (config, ds), (_, path) in sorted(best.items())]


def load_run_metrics(
    config_name: str,
    dataset: str,
    metrics_path: Path,
) -> dict:
    """Load metrics + optional faithfulness metrics into a flat row dict."""
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    # Look for faithfulness_metrics.json in the same directory
    faithfulness_path = metrics_path.parent / "faithfulness_metrics.json"
    faith_metrics: dict = {}
    if faithfulness_path.exists():
        with open(faithfulness_path, encoding="utf-8") as f:
            faith_metrics = json.load(f)

    row: dict = {"Config": config_name, "Dataset": dataset}
    for col in METRIC_COLUMNS:
        if col in ("Config", "Dataset"):
            continue
        if col in ("AvgFaithfulness", "HallucinationRate"):
            row[col] = faith_metrics.get(col, "")
        else:
            row[col] = metrics.get(col, "")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Phase 4 experiment results.")
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "experiments" / "phase4_results.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "experiments" / "phase4_results.json",
    )
    args = parser.parse_args()

    runs = discover_runs(args.matrix_dir)
    if not runs:
        print(f"No metrics.json files found under {args.matrix_dir}")
        sys.exit(1)

    print(f"Found {len(runs)} experiment run(s).")

    rows = []
    for config_name, dataset, metrics_path in runs:
        row = load_run_metrics(config_name, dataset, metrics_path)
        rows.append(row)
        em = row.get('EM', '?')
        f1 = row.get('F1', '?')
        em_str = f"{em:.3f}" if isinstance(em, (int, float)) else str(em)
        f1_str = f"{f1:.3f}" if isinstance(f1, (int, float)) else str(f1)
        print(f"  {config_name:30s} {dataset:12s} EM={em_str:>6s} F1={f1_str:>6s}")

    # Write CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV -> {args.output_csv}")

    # Write JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
