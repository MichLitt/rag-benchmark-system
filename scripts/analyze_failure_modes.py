"""Failure mode analysis CLI.

Reads a predictions.json file (produced by run_naive_rag_baseline.py) and
classifies each record into Hotpot-oriented retrieval failure buckets plus
generation failure.

Usage:
  python scripts/analyze_failure_modes.py \
    --predictions experiments/runs/<RUN>/hotpotqa/predictions.json \
    --corpus-path data/raw/corpus/wiki_passages/passages.jsonl.gz \
    --output-dir experiments/reports/failure_modes/<RUN>/hotpotqa

The --corpus-path argument is optional. Without it, coverage_failure cannot be
detected and those records will be classified as recall_failure instead.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.failure_mode import classify_all, summarize
from src.corpus import iter_corpus_documents
from src.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify RAG predictions into failure modes.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.json from a run_naive_rag_baseline.py run.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Optional corpus JSONL/JSONL.GZ path for coverage failure detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write summary.json and details.json. Defaults to predictions file dir.",
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=0.3,
        help="F1 score threshold above which a prediction is considered correct (default 0.3).",
    )
    parser.add_argument(
        "--good-rank-cutoff",
        type=int,
        default=2,
        help="How many top retrieved titles count as 'well ranked' (default 2).",
    )
    parser.add_argument(
        "--max-corpus-docs",
        type=int,
        default=0,
        help="Limit corpus scan to first N docs. <=0 means full corpus.",
    )
    return parser.parse_args()


def _normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"[_-]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title


def _build_corpus_titles(corpus_path: Path, max_docs: int) -> set[str]:
    titles: set[str] = set()
    for i, doc in enumerate(iter_corpus_documents(corpus_path)):
        if doc.title:
            titles.add(_normalize_title(doc.title))
        if max_docs > 0 and i + 1 >= max_docs:
            break
    print(f"Corpus titles loaded: {len(titles):,} unique titles from {corpus_path.name}")
    return titles


def main() -> None:
    args = parse_args()

    with args.predictions.open("r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} prediction records from {args.predictions}")

    corpus_titles = None
    if args.corpus_path is not None:
        corpus_titles = _build_corpus_titles(args.corpus_path, int(args.max_corpus_docs))

    results = classify_all(
        predictions,
        corpus_titles=corpus_titles,
        f1_threshold=args.f1_threshold,
        good_rank_cutoff=args.good_rank_cutoff,
    )
    summary = summarize(results)

    out_dir = ensure_dir(args.output_dir or args.predictions.parent)

    summary_path = out_dir / "failure_mode_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    details = [
        {
            "query_id": r.query_id,
            "failure_mode": r.failure_mode.value,
            "f1": r.f1,
            "is_em": r.is_em,
            "recall_at_k": r.recall_at_k,
            "predicted_answer": r.predicted_answer,
            "gold_answers": r.gold_answers,
            "gold_titles": r.gold_titles,
            "retrieved_titles": r.retrieved_titles,
        }
        for r in results
    ]
    details_path = out_dir / "failure_mode_details.json"
    with details_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
