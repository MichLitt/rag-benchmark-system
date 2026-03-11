from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.hotpot_failure_taxonomy import (
    build_markdown_report,
    classify_taxonomy_record,
    load_json,
    merge_qa_fields,
    records_to_dicts,
    resolve_query_id,
    summarize_taxonomy,
    write_json,
    write_jsonl,
)
from src.flashrag_loader import load_flashrag_qa
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.postprocess import normalize_title
from src.retrieval.sharded_dense import ShardedFaissDenseRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Hotpot retrieval failures from details.json.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing details.json and metrics.json.")
    parser.add_argument("--details-path", type=Path, default=None, help="Explicit details.json path.")
    parser.add_argument("--metrics-path", type=Path, default=None, help="Explicit metrics.json path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Dense sharded manifest for flat probe retrieval.")
    parser.add_argument("--qa-path", type=Path, default=None, help="Optional qa.jsonl path to backfill missing fields.")
    parser.add_argument(
        "--probe-top-k",
        type=int,
        default=300,
        help="Dense probe depth used to detect budget-limited misses.",
    )
    parser.add_argument(
        "--title-bm25-manifest",
        type=Path,
        default=None,
        help="Title-only BM25 manifest used for sparse lexical probes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write taxonomy_summary.json and taxonomy_examples.jsonl. Defaults next to details.json.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Markdown report output path. Defaults under report/ with today's date.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.run_dir is not None:
        details_path = args.run_dir / "details.json"
        metrics_path = args.metrics_path or (args.run_dir / "metrics.json")
    elif args.details_path is not None:
        details_path = args.details_path
        metrics_path = args.metrics_path
    else:
        raise ValueError("Provide either --run-dir or --details-path")
    if not details_path.exists():
        raise FileNotFoundError(f"details file does not exist: {details_path}")
    if metrics_path is not None and not metrics_path.exists():
        metrics_path = None
    return details_path, metrics_path


def _load_title_probe_retriever(title_manifest_path: Path | None) -> BM25Retriever | None:
    if title_manifest_path is None:
        return None
    payload = load_json(title_manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid title manifest: {title_manifest_path}")
    return BM25Retriever(
        bm25_path=payload["bm25_path"],
        docstore_path=payload["docstore_path"],
    )


def main() -> None:
    args = parse_args()
    details_path, metrics_path = _resolve_paths(args)
    details = load_json(details_path)
    if not isinstance(details, list):
        raise ValueError(f"Expected a JSON array in {details_path}")
    metrics = load_json(metrics_path) if metrics_path is not None else {}
    if metrics and not isinstance(metrics, dict):
        raise ValueError(f"Expected a JSON object in {metrics_path}")

    if args.qa_path is not None:
        qa_samples = load_flashrag_qa(args.qa_path)
        qa_rows = [
            {"id": sample.query_id, "question": sample.question, "gold_titles": sample.gold_titles}
            for sample in qa_samples
        ]
        details = merge_qa_fields(details, qa_rows)

    dense_retriever = None
    if args.manifest is not None:
        dense_retriever = ShardedFaissDenseRetriever(args.manifest)
    title_retriever = _load_title_probe_retriever(args.title_bm25_manifest)

    details_by_id = {resolve_query_id(record): dict(record) for record in details if resolve_query_id(record)}
    probe_targets = [
        record
        for record in details
        if str(record.get("retrieval_failure_bucket", "")) in {"no_gold_in_raw", "only_one_gold_in_raw"}
    ]

    dense_probe_titles_by_id: dict[str, list[str]] = {}
    sparse_probe_titles_by_id: dict[str, list[str]] = {}

    if dense_retriever is not None and probe_targets:
        questions = [str(record.get("question", "")) for record in probe_targets]
        dense_probe_results = dense_retriever.retrieve_many(questions, top_k=int(args.probe_top_k))
        for record, docs in zip(probe_targets, dense_probe_results):
            dense_probe_titles_by_id[resolve_query_id(record)] = [doc.title for doc in docs if doc.title.strip()]

    if title_retriever is not None and probe_targets:
        for record in tqdm(probe_targets, desc="Sparse probing title BM25"):
            docs = title_retriever.retrieve(str(record.get("question", "")), top_k=50)
            sparse_probe_titles_by_id[resolve_query_id(record)] = [doc.title for doc in docs if doc.title.strip()]

    taxonomy_records = [
        classify_taxonomy_record(
            details_by_id[resolve_query_id(record)],
            dense_probe_titles=dense_probe_titles_by_id.get(resolve_query_id(record), []),
            sparse_probe_titles=sparse_probe_titles_by_id.get(resolve_query_id(record), []),
        )
        for record in details
        if resolve_query_id(record) in details_by_id
    ]

    summary = summarize_taxonomy(taxonomy_records, metrics=metrics if isinstance(metrics, dict) else {})
    summary["probe_top_k"] = int(args.probe_top_k)
    summary["title_probe_top_k"] = 50
    summary["details_path"] = str(details_path.resolve())
    summary["metrics_path"] = str(metrics_path.resolve()) if metrics_path is not None else ""
    summary["manifest_path"] = str(args.manifest.resolve()) if args.manifest is not None else ""
    summary["title_bm25_manifest_path"] = (
        str(args.title_bm25_manifest.resolve()) if args.title_bm25_manifest is not None else ""
    )

    output_dir = args.output_dir or details_path.parent / "failure_taxonomy"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "taxonomy_summary.json"
    examples_path = output_dir / "taxonomy_examples.jsonl"
    report_path = args.report_path or Path("report") / f"hotpot_failure_taxonomy_{datetime.now():%Y%m%d}.md"

    write_json(summary_path, summary)
    write_jsonl(examples_path, records_to_dicts(taxonomy_records))
    report_text = build_markdown_report(
        summary=summary,
        records=taxonomy_records,
        details_path=details_path,
        metrics_path=metrics_path,
        manifest_path=args.manifest,
        title_bm25_manifest_path=args.title_bm25_manifest,
        probe_top_k=int(args.probe_top_k),
        title_probe_top_k=50,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    if dense_retriever is not None:
        dense_retriever.close()

    print(
        json.dumps(
            {
                "summary_path": str(summary_path.resolve()),
                "examples_path": str(examples_path.resolve()),
                "report_path": str(report_path.resolve()),
                "top_blockers": summary["top_blockers"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
