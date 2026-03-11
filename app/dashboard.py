"""RAG Benchmark Study — Experiment Dashboard.

Streamlit app that displays pre-computed experiment results, failure
analysis, and case studies from the Phase 4 matrix.

Usage:
    uv run streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BUNDLE_PATH = Path(__file__).parent / "data" / "dashboard_bundle.json"


@st.cache_data
def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        return {}
    with open(BUNDLE_PATH) as f:
        return json.load(f)


def render_results_table(results: list[dict]) -> None:
    st.header("Experiment Results")
    if not results:
        st.warning("No results data available. Run the experiment matrix first.")
        return

    df = pd.DataFrame(results)
    display_cols = [
        "Config", "Dataset", "NumQueries", "EM", "F1", "Recall@k",
        "AvgFaithfulness", "HallucinationRate",
        "AvgLatencyMs", "TotalGenerationCostUsd", "GenerationErrorRate",
    ]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].style.format(
            {
                "EM": "{:.3f}",
                "F1": "{:.3f}",
                "Recall@k": "{:.3f}",
                "AvgFaithfulness": "{:.3f}",
                "HallucinationRate": "{:.3f}",
                "AvgLatencyMs": "{:.0f}",
                "TotalGenerationCostUsd": "{:.4f}",
                "GenerationErrorRate": "{:.3f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_accuracy_latency_scatter(results: list[dict]) -> None:
    st.header("Accuracy vs Latency")
    if not results:
        return

    df = pd.DataFrame(results)
    required = {"F1", "AvgLatencyMs", "Config", "Dataset"}
    if not required.issubset(df.columns):
        st.warning("Missing columns for scatter plot.")
        return

    df = df.dropna(subset=["F1", "AvgLatencyMs"])
    if df.empty:
        st.info("No data with both F1 and latency.")
        return

    fig = px.scatter(
        df,
        x="AvgLatencyMs",
        y="F1",
        color="Config",
        symbol="Dataset",
        hover_data=["EM", "Recall@k", "TotalGenerationCostUsd"],
        labels={
            "AvgLatencyMs": "Avg Total Latency (ms)",
            "F1": "F1 Score",
        },
        title="F1 vs Avg Latency by Configuration",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_cost_comparison(results: list[dict]) -> None:
    st.header("Cost Comparison")
    if not results:
        return

    df = pd.DataFrame(results)
    if "TotalGenerationCostUsd" not in df.columns or "Config" not in df.columns:
        return

    df = df.dropna(subset=["TotalGenerationCostUsd"])
    if df.empty:
        return

    fig = px.bar(
        df,
        x="Config",
        y="TotalGenerationCostUsd",
        color="Dataset",
        barmode="group",
        labels={"TotalGenerationCostUsd": "Total Cost (USD)"},
        title="Generation Cost by Configuration",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_failure_modes(failure_analysis: dict) -> None:
    st.header("Failure Mode Breakdown")
    if not failure_analysis or "results" not in failure_analysis:
        st.warning("No failure analysis data available.")
        return

    results = failure_analysis["results"]
    rows = []
    for key, data in results.items():
        config, dataset = key.split("|", 1) if "|" in key else (key, "unknown")
        summary = data.get("summary", {})
        total = summary.get("total", 0)
        for mode_name, mode_data in summary.items():
            if mode_name == "total" or not isinstance(mode_data, dict):
                continue
            rows.append({
                "Config": config,
                "Dataset": dataset,
                "FailureMode": mode_name,
                "Count": mode_data.get("count", 0),
                "Pct": mode_data.get("pct", 0),
            })

    if not rows:
        st.info("No failure mode data to display.")
        return

    df = pd.DataFrame(rows)

    # Per-dataset stacked bar
    datasets = df["Dataset"].unique()
    for dataset in sorted(datasets):
        subset = df[df["Dataset"] == dataset]
        fig = px.bar(
            subset,
            x="Config",
            y="Pct",
            color="FailureMode",
            title=f"Failure Mode Distribution — {dataset}",
            labels={"Pct": "Percentage (%)"},
        )
        fig.update_layout(height=400, barmode="stack")
        st.plotly_chart(fig, use_container_width=True)


def render_case_studies(case_studies: list[dict], sample_predictions: dict) -> None:
    st.header("Case Study Viewer")
    if not case_studies and not sample_predictions:
        st.info("No case study data available.")
        return

    if case_studies:
        st.subheader("Highlighted Cases")
        categories = sorted(set(c["category"] for c in case_studies))
        selected_cat = st.selectbox("Category", categories)
        filtered = [c for c in case_studies if c["category"] == selected_cat]

        for case in filtered:
            with st.expander(f"{case['dataset']} — {case['query_id']}: {case.get('question', '')[:80]}..."):
                st.write(f"**Question:** {case.get('question', '')}")
                st.write(f"**Gold answers:** {', '.join(case.get('gold_answers', []))}")
                configs_data = case.get("configs", {})
                cols = st.columns(len(configs_data))
                for col, (config, data) in zip(cols, configs_data.items()):
                    with col:
                        st.markdown(f"**{config}**")
                        st.write(f"Answer: {data.get('predicted_answer', '')}")
                        st.write(f"F1: {data.get('f1', 0):.3f} | EM: {data.get('is_em', False)}")
                        titles = data.get("retrieved_titles", [])
                        if titles:
                            st.write(f"Top titles: {', '.join(titles[:3])}")

    if sample_predictions:
        st.subheader("Browse Predictions")
        combos = sorted(sample_predictions.keys())
        selected = st.selectbox("Config | Dataset", combos)
        preds = sample_predictions.get(selected, [])
        if preds:
            df = pd.DataFrame(preds)
            display = ["query_id", "question", "predicted_answer", "gold_answers", "f1", "is_em"]
            available = [c for c in display if c in df.columns]
            st.dataframe(df[available], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="RAG Benchmark Dashboard",
        layout="wide",
    )
    st.title("RAG Benchmark Study — Experiment Dashboard")

    bundle = load_bundle()
    if not bundle:
        st.error(
            f"Dashboard data not found at `{BUNDLE_PATH}`. "
            "Run the experiment matrix and export script first:\n\n"
            "```\n"
            "uv run python scripts/run_phase4_matrix.py\n"
            "uv run python scripts/aggregate_experiment_results.py --matrix-dir experiments/runs/phase4_matrix/\n"
            "uv run python scripts/export_dashboard_data.py --matrix-dir experiments/runs/phase4_matrix/\n"
            "```"
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Results Table",
        "Charts",
        "Failure Modes",
        "Case Studies",
    ])

    with tab1:
        render_results_table(bundle.get("results", []))

    with tab2:
        render_accuracy_latency_scatter(bundle.get("results", []))
        render_cost_comparison(bundle.get("results", []))

    with tab3:
        render_failure_modes(bundle.get("failure_analysis", {}))

    with tab4:
        render_case_studies(
            bundle.get("case_studies", []),
            bundle.get("sample_predictions", {}),
        )


if __name__ == "__main__":
    main()
