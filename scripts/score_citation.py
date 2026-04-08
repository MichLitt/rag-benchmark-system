"""CLI: batch-score citation metrics on an existing run-result JSONL file.

Reads a run-result JSONL (fields: query_id, predicted_answer, retrieved_texts,
optionally gold_page_set), computes answer_attribution_rate /
supporting_passage_hit / page_grounding_accuracy for each row using HHEM, and
writes an augmented JSONL to --output.

Usage
-----
    uv run python scripts/score_citation.py \\
        --input  results/run_results.jsonl \\
        --output results/run_results_cited.jsonl \\
        --device cpu

The script skips rows where retrieved_texts or predicted_answer is absent/empty.
Rows that already have citation scores are overwritten.

Note: Downloads ~430 MB from HuggingFace on first run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.citation import CitationEvaluator
from src.evaluation.hhem_scorer import HHEMScorer
from src.logging_utils import configure_logging, get_logger
from src.types import Document


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score citation metrics on a run-result JSONL.")
    p.add_argument("--input", required=True, type=Path, help="Input run-result JSONL.")
    p.add_argument("--output", required=True, type=Path, help="Output augmented JSONL path.")
    p.add_argument("--device", default=None, help="Torch device (cpu / cuda). Auto-detected.")
    p.add_argument("--threshold", type=float, default=0.5, help="NLI entailment threshold.")
    p.add_argument("--page-tolerance", type=int, default=1, help="Page range tolerance (±N).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    rows = _load_rows(args.input)
    logger.info("Loaded %d rows from %s", len(rows), args.input)

    logger.info("Loading HHEM model (this may take a moment on first run) …")
    scorer = HHEMScorer(device=args.device)
    evaluator = CitationEvaluator(
        scorer, threshold=args.threshold, page_tolerance=args.page_tolerance
    )

    skipped = 0
    for i, row in enumerate(rows):
        answer = str(row.get("predicted_answer") or "").strip()
        retrieved_texts: list[str] = row.get("retrieved_texts") or []

        if not answer or not retrieved_texts:
            skipped += 1
            continue

        passages = [
            Document(doc_id=f"ctx{j}", text=t, title="")
            for j, t in enumerate(retrieved_texts)
        ]

        # gold_page_set: list[int] in JSON → set[int]
        raw_pages = row.get("gold_page_set")
        gold_pages = set(map(int, raw_pages)) if raw_pages else None

        result = evaluator.evaluate(answer, passages, gold_page_set=gold_pages)

        row["answer_attribution_rate"] = result.answer_attribution_rate
        row["supporting_passage_hit"] = result.supporting_passage_hit
        row["page_grounding_accuracy"] = result.page_grounding_accuracy

        if (i + 1) % 10 == 0:
            logger.info("Scored %d/%d rows …", i + 1, len(rows))

    if skipped:
        logger.warning("Skipped %d rows (missing answer or retrieved_texts).", skipped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Written %d rows to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
