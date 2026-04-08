#!/usr/bin/env python3
"""Batch NLI citation scoring for existing run result JSON files.

Reads a run results JSON (produced by run_naive_rag_baseline.py or
run_phase4_matrix.py), scores each row's retrieved_texts against the
predicted_answer using HHEM, and writes the enriched results to a new JSON.

Usage:
    uv run python scripts/score_citation.py \\
        --input experiments/phase4_results.json \\
        --output /tmp/phase4_results_scored.json

Note: Requires transformers and torch. First run will download the HHEM model
(~500 MB). Set HF_HOME to control the cache location.
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
from src.types import Document


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch NLI citation scoring for RAG run results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path, required=True,
        help="Run results JSON file (list of result dicts).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output JSON path with nli_* fields appended to each row.",
    )
    p.add_argument(
        "--hhem-model",
        default=HHEMScorer.MODEL_NAME,
        help="HuggingFace model name for HHEM.",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Consistency threshold for is_consistent flag.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading HHEM model: {args.hhem_model}")
    scorer = HHEMScorer(model_name=args.hhem_model, threshold=args.threshold)
    evaluator = CitationEvaluator(scorer)

    with args.input.open("r", encoding="utf-8") as f:
        rows: list[dict] = json.load(f)

    print(f"Scoring {len(rows)} rows...")
    enriched: list[dict] = []
    for i, row in enumerate(rows, start=1):
        answer = row.get("predicted_answer", "")
        texts: list[str] = row.get("retrieved_texts", [])
        # Build lightweight Documents from retrieved_texts (no page metadata
        # in legacy run results — page_grounding_accuracy will be None).
        passages = [
            Document(doc_id=f"p{j}", text=t, title="")
            for j, t in enumerate(texts)
        ]
        result = evaluator.evaluate(answer, passages)
        row = dict(row)  # shallow copy to avoid mutating original
        row["nli_answer_attribution_rate"] = result.answer_attribution_rate
        row["nli_supporting_passage_hit"] = result.supporting_passage_hit
        row["nli_page_grounding_accuracy"] = result.page_grounding_accuracy
        enriched.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} scored")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(enriched)} rows → {args.output}")

    # Summary stats
    rates = [r["nli_answer_attribution_rate"] for r in enriched]
    hits = [r["nli_supporting_passage_hit"] for r in enriched]
    avg_rate = sum(rates) / len(rates) if rates else 0.0
    hit_rate = sum(hits) / len(hits) if hits else 0.0
    print(f"avg nli_answer_attribution_rate: {avg_rate:.3f}")
    print(f"supporting_passage_hit rate:     {hit_rate:.3f}")


if __name__ == "__main__":
    main()
