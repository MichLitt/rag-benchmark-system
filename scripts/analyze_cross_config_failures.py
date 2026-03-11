"""Cross-config failure comparison across the Phase 4 experiment matrix.

Classifies predictions using HotpotQA-specific or generic failure modes,
then generates a comparison table and identifies flip cases.

Usage:
    uv run python scripts/analyze_cross_config_failures.py \
        --matrix-dir experiments/runs/phase4_matrix/
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

from src.analysis.failure_mode import (
    FailureMode,
    classify_all as classify_hotpot,
    summarize as summarize_hotpot,
)
from src.analysis.generic_failure_mode import (
    GenericFailureMode,
    classify_all as classify_generic,
    summarize as summarize_generic,
)

HOTPOT_DATASETS = {"hotpotqa"}


def discover_predictions(matrix_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover (config_name, dataset, predictions_path) tuples."""
    results = []
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
            # Try metrics.json in same dir
            metrics_path = pred_path.parent / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as f:
                    dataset = json.load(f).get("Dataset", "unknown")
            else:
                dataset = "unknown"
        results.append((config_name, dataset, pred_path))
    return results


def classify_predictions(dataset: str, predictions: list[dict]) -> dict:
    """Classify predictions and return summary."""
    if dataset in HOTPOT_DATASETS:
        results = classify_hotpot(predictions)
        return {
            "classifier": "hotpotqa",
            "summary": summarize_hotpot(results),
            "per_query": {r.query_id: r.failure_mode.value for r in results},
        }
    else:
        results = classify_generic(predictions)
        return {
            "classifier": "generic",
            "summary": summarize_generic(results),
            "per_query": {r.query_id: r.failure_mode.value for r in results},
        }


def find_flip_cases(
    all_results: dict[tuple[str, str], dict],
) -> list[dict]:
    """Find queries where configs disagree on correctness."""
    # Group by dataset
    by_dataset: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for (config, dataset), result in all_results.items():
        by_dataset[dataset][config] = result["per_query"]

    flips = []
    for dataset, config_queries in by_dataset.items():
        configs = list(config_queries.keys())
        if len(configs) < 2:
            continue
        all_query_ids = set()
        for q in config_queries.values():
            all_query_ids.update(q.keys())

        for qid in sorted(all_query_ids):
            modes = {}
            for config in configs:
                modes[config] = config_queries[config].get(qid, "missing")
            correct_configs = [c for c, m in modes.items() if m == "correct"]
            wrong_configs = [c for c, m in modes.items() if m != "correct" and m != "missing"]
            if correct_configs and wrong_configs:
                flips.append({
                    "dataset": dataset,
                    "query_id": qid,
                    "correct_in": correct_configs,
                    "failed_in": wrong_configs,
                    "modes": modes,
                })
    return flips


def generate_markdown_table(
    all_results: dict[tuple[str, str], dict],
) -> str:
    """Generate a markdown comparison table."""
    lines = []

    # Group by dataset
    by_dataset: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (config, dataset), result in sorted(all_results.items()):
        by_dataset[dataset].append((config, result))

    for dataset, config_results in sorted(by_dataset.items()):
        lines.append(f"\n### {dataset}\n")

        # Get all failure mode names for this dataset
        classifier = config_results[0][1]["classifier"]
        if classifier == "hotpotqa":
            mode_names = [m.value for m in FailureMode]
        else:
            mode_names = [m.value for m in GenericFailureMode]

        # Header
        header = "| Failure Mode |"
        separator = "| --- |"
        for config, _ in config_results:
            header += f" {config} |"
            separator += " ---: |"
        lines.append(header)
        lines.append(separator)

        # Rows
        for mode_name in mode_names:
            row = f"| {mode_name} |"
            for _, result in config_results:
                summary = result["summary"]
                entry = summary.get(mode_name, {})
                if isinstance(entry, dict):
                    pct = entry.get("pct", 0)
                    count = entry.get("count", 0)
                    row += f" {pct}% ({count}) |"
                else:
                    row += f" - |"
            lines.append(row)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-config failure comparison.")
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "report" / "failure_comparison_table.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "experiments" / "phase4_failure_analysis.json",
    )
    args = parser.parse_args()

    discoveries = discover_predictions(args.matrix_dir)
    if not discoveries:
        print(f"No predictions.json files found under {args.matrix_dir}")
        sys.exit(1)

    print(f"Found {len(discoveries)} prediction file(s).\n")

    all_results: dict[tuple[str, str], dict] = {}
    for config_name, dataset, pred_path in discoveries:
        print(f"  Classifying: {config_name} / {dataset}")
        with open(pred_path, encoding="utf-8") as f:
            predictions = json.load(f)
        if not isinstance(predictions, list):
            print(f"    SKIP: not a list")
            continue
        result = classify_predictions(dataset, predictions)
        all_results[(config_name, dataset)] = result

    # Generate markdown report
    md = "# Failure Mode Comparison — Phase 4 Matrix\n"
    md += generate_markdown_table(all_results)

    # Flip cases
    flips = find_flip_cases(all_results)
    md += f"\n\n## Flip Cases ({len(flips)} queries)\n\n"
    md += "Queries where at least one config succeeded and another failed:\n\n"
    for flip in flips[:20]:
        md += f"- **{flip['dataset']}** `{flip['query_id']}`: "
        md += f"correct in {flip['correct_in']}, failed in {flip['failed_in']}\n"
    if len(flips) > 20:
        md += f"\n... and {len(flips) - 20} more.\n"

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nReport -> {args.output_report}")

    # Write structured JSON
    json_data = {
        "results": {
            f"{config}|{dataset}": {
                "classifier": result["classifier"],
                "summary": result["summary"],
            }
            for (config, dataset), result in all_results.items()
        },
        "flips": flips,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
