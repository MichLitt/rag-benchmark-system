"""Post-hoc faithfulness scoring for existing experiment predictions.

Reads a predictions.json, scores each record via LLM-as-judge,
and writes faithfulness.json + faithfulness_metrics.json alongside.

Usage:
    uv run python scripts/score_faithfulness.py \
        --predictions-path experiments/runs/phase4_matrix/C2_dense_rerank/hotpotqa/predictions.json \
        --max-queries 10

    # Score all predictions in a matrix directory
    uv run python scripts/score_faithfulness.py \
        --matrix-dir experiments/runs/phase4_matrix/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.evaluation.faithfulness import FaithfulnessResult, score_faithfulness

load_dotenv(ROOT / ".env")


def find_predictions_files(matrix_dir: Path) -> list[Path]:
    """Find all predictions.json files under a matrix directory."""
    results = []
    for path in sorted(matrix_dir.rglob("predictions.json")):
        results.append(path)
    return results


def score_predictions_file(
    predictions_path: Path,
    *,
    api_key: str,
    api_base: str,
    model: str,
    max_completion_tokens: int,
    max_queries: int,
    timeout_sec: int,
) -> None:
    """Score faithfulness for a single predictions.json file."""
    output_dir = predictions_path.parent
    faithfulness_path = output_dir / "faithfulness.json"
    metrics_path = output_dir / "faithfulness_metrics.json"

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, list):
        print(f"  SKIP: {predictions_path} is not a list")
        return

    if max_queries > 0:
        predictions = predictions[:max_queries]

    total = len(predictions)
    results: list[dict] = []
    scores: list[float] = []
    errors = 0

    print(f"  Scoring {total} predictions...")

    for i, record in enumerate(predictions):
        question = record.get("question", record.get("query", ""))
        answer = record.get("predicted_answer", "")
        # Build context from retrieved titles + chunks stored in prediction
        context_texts: list[str] = []
        retrieved_titles = record.get("retrieved_titles", [])
        retrieved_texts = record.get("retrieved_texts", [])
        if retrieved_texts:
            for title, text in zip(retrieved_titles, retrieved_texts):
                context_texts.append(f"Title: {title}\n{text}")
        elif retrieved_titles:
            # Fallback: just use titles if texts aren't stored
            context_texts = [f"Title: {t}" for t in retrieved_titles]

        result = score_faithfulness(
            question=question,
            answer=answer,
            context_texts=context_texts,
            api_key=api_key,
            api_base=api_base,
            model=model,
            max_completion_tokens=max_completion_tokens,
            timeout_sec=timeout_sec,
        )

        scores.append(result.score)
        if result.error:
            errors += 1

        results.append({
            "query_id": record.get("query_id", i),
            "faithfulness_score": result.score,
            "reasoning": result.reasoning,
            "error": result.error,
        })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            avg = sum(scores) / len(scores) if scores else 0.0
            print(f"    [{i+1}/{total}] avg={avg:.3f} errors={errors}")

    # Write per-record faithfulness scores
    with open(faithfulness_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Write aggregate metrics
    avg_score = sum(scores) / len(scores) if scores else 0.0
    hallucination_rate = sum(1 for s in scores if s < 0.5) / len(scores) if scores else 0.0
    metrics = {
        "AvgFaithfulness": round(avg_score, 4),
        "HallucinationRate": round(hallucination_rate, 4),
        "NumScored": len(scores),
        "NumScoringErrors": errors,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  -> {faithfulness_path}")
    print(f"  -> {metrics_path}")
    print(f"  AvgFaithfulness={avg_score:.3f}, HallucinationRate={hallucination_rate:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score faithfulness on experiment predictions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions-path", type=Path, help="Path to a single predictions.json.")
    group.add_argument("--matrix-dir", type=Path, help="Root dir to find all predictions.json files.")
    parser.add_argument("--max-queries", type=int, default=0, help="Limit per file (0 = all).")
    parser.add_argument("--model", type=str, default="MiniMax-M2.5")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout-sec", type=int, default=30)
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY", "")
    api_base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        print("ERROR: LLM_API_KEY environment variable not set.")
        sys.exit(1)

    if args.predictions_path:
        files = [args.predictions_path]
    else:
        files = find_predictions_files(args.matrix_dir)

    if not files:
        print("No predictions.json files found.")
        sys.exit(1)

    print(f"Found {len(files)} predictions file(s) to score.\n")

    for path in files:
        print(f"Scoring: {path}")
        start = time.time()
        score_predictions_file(
            path,
            api_key=api_key,
            api_base=api_base,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            max_queries=args.max_queries,
            timeout_sec=args.timeout_sec,
        )
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
