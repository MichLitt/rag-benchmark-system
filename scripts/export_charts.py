"""Export static chart images for the README.

Reads the aggregated experiment results and generates PNG charts
using Plotly + kaleido.

Usage:
    uv run python scripts/export_charts.py
    uv run python scripts/export_charts.py --results-json experiments/phase4_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static charts as PNG.")
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
        "--output-dir",
        type=Path,
        default=ROOT / "report" / "charts",
    )
    args = parser.parse_args()

    # Lazy import to allow running without plotly installed in test envs
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Results data ---
    if not args.results_json.exists():
        print(f"Results JSON not found: {args.results_json}")
        sys.exit(1)

    with open(args.results_json, encoding="utf-8") as f:
        results = json.load(f)
    df = pd.DataFrame(results)

    # Chart 1: Accuracy vs Latency scatter
    if {"F1", "AvgLatencyMs", "Config", "Dataset"}.issubset(df.columns):
        subset = df.dropna(subset=["F1", "AvgLatencyMs"])
        if not subset.empty:
            fig = px.scatter(
                subset,
                x="AvgLatencyMs",
                y="F1",
                color="Config",
                symbol="Dataset",
                labels={
                    "AvgLatencyMs": "Avg Total Latency (ms)",
                    "F1": "F1 Score",
                },
                title="F1 vs Avg Latency by Configuration",
            )
            fig.update_layout(width=900, height=500)
            path = args.output_dir / "accuracy_vs_latency.png"
            fig.write_image(str(path))
            print(f"  -> {path}")

    # Chart 2: Cost comparison
    if {"TotalGenerationCostUsd", "Config", "Dataset"}.issubset(df.columns):
        subset = df.dropna(subset=["TotalGenerationCostUsd"])
        if not subset.empty:
            fig = px.bar(
                subset,
                x="Config",
                y="TotalGenerationCostUsd",
                color="Dataset",
                barmode="group",
                labels={"TotalGenerationCostUsd": "Total Cost (USD)"},
                title="Generation Cost by Configuration",
            )
            fig.update_layout(width=900, height=400)
            path = args.output_dir / "cost_comparison.png"
            fig.write_image(str(path))
            print(f"  -> {path}")

    # Chart 3: Failure mode breakdown
    if args.failure_json.exists():
        with open(args.failure_json, encoding="utf-8") as f:
            failure_data = json.load(f)

        fa_results = failure_data.get("results", {})
        rows = []
        for key, data in fa_results.items():
            config, dataset = key.split("|", 1) if "|" in key else (key, "unknown")
            summary = data.get("summary", {})
            for mode_name, mode_data in summary.items():
                if mode_name == "total" or not isinstance(mode_data, dict):
                    continue
                rows.append({
                    "Config": config,
                    "Dataset": dataset,
                    "FailureMode": mode_name,
                    "Pct": mode_data.get("pct", 0),
                })

        if rows:
            fdf = pd.DataFrame(rows)
            fig = px.bar(
                fdf,
                x="Config",
                y="Pct",
                color="FailureMode",
                facet_col="Dataset",
                labels={"Pct": "Percentage (%)"},
                title="Failure Mode Distribution",
            )
            fig.update_layout(width=1200, height=400, barmode="stack")
            path = args.output_dir / "failure_modes.png"
            fig.write_image(str(path))
            print(f"  -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
