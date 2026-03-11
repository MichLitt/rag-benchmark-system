from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_yaml_config
from src.flashrag_loader import load_flashrag_qa
from src.io_utils import ensure_dir, save_json
from src.query import build_query_expander


DEFAULT_CONFIG_PATH = Path("config/wiki18_21m_sharded_hotpot_screening_decompose_noorig3.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay failed Hotpot query expansion samples.")
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--qa-path", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_qa_path(predictions_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    metrics_path = predictions_path.with_name("metrics.json")
    if metrics_path.exists():
        payload = _load_json(metrics_path)
        if isinstance(payload, dict):
            qa_path = str(payload.get("QAPath", "")).strip()
            if qa_path:
                return Path(qa_path)
    raise ValueError("Unable to resolve qa path. Provide --qa-path explicitly.")


def _load_failed_rows(predictions_path: Path, max_examples: int) -> list[dict]:
    payload = _load_json(predictions_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {predictions_path}")
    failed = [row for row in payload if str(row.get("query_expansion_error", "")).strip()]
    if max_examples > 0:
        return failed[:max_examples]
    return failed


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions_path
    qa_path = _resolve_qa_path(predictions_path, args.qa_path)
    cfg = load_yaml_config(args.config)
    expander = build_query_expander(cfg, dataset_name=args.dataset, mode_override="hotpot_decompose")
    if expander is None:
        raise RuntimeError("Config did not produce a hotpot_decompose expander.")

    samples = load_flashrag_qa(qa_path)
    qa_by_id = {sample.query_id: sample for sample in samples}
    failed_rows = _load_failed_rows(predictions_path, int(args.max_examples))

    output_dir = args.output_dir or predictions_path.parent / f"decompose_replay_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = ensure_dir(output_dir)

    replay_results: list[dict] = []
    failure_reason_counter: Counter[str] = Counter()
    request_error_counter: Counter[str] = Counter()
    parsed_without_fallback = 0
    fallback_used = 0
    usable_queries = 0

    for row in failed_rows:
        query_id = str(row.get("query_id", "")).strip()
        sample = qa_by_id.get(query_id)
        if sample is None:
            raise KeyError(f"Unable to find query_id {query_id} in {qa_path}")
        try:
            queries = expander.expand_queries(sample.question)
            metadata = {}
            getter = getattr(expander, "get_last_expansion_metadata", None)
            if callable(getter):
                payload = getter()
                if isinstance(payload, dict):
                    metadata = payload
            failure_reason = str(metadata.get("failure_reason", "")).strip()
            used_fallback = bool(metadata.get("used_fallback", False))
            if failure_reason:
                failure_reason_counter[failure_reason] += 1
            if used_fallback:
                fallback_used += 1
            else:
                parsed_without_fallback += 1
            usable_queries += 1
            replay_results.append(
                {
                    "query_id": query_id,
                    "question": sample.question,
                    "old_query_expansion_error": row.get("query_expansion_error", ""),
                    "new_queries": queries,
                    "query_count": len(queries),
                    "used_fallback": used_fallback,
                    "failure_reason": failure_reason,
                    "cache_key": str(metadata.get("cache_key", "")).strip(),
                    "salvage_stage": str(metadata.get("salvage_stage", "")).strip(),
                    "request_error": "",
                }
            )
        except Exception as exc:
            request_error = f"{type(exc).__name__}: {exc}"
            request_error_counter[type(exc).__name__] += 1
            replay_results.append(
                {
                    "query_id": query_id,
                    "question": sample.question,
                    "old_query_expansion_error": row.get("query_expansion_error", ""),
                    "new_queries": [],
                    "query_count": 0,
                    "used_fallback": False,
                    "failure_reason": "",
                    "cache_key": "",
                    "salvage_stage": "",
                    "request_error": request_error,
                }
            )

    total = len(failed_rows)
    summary = {
        "predictions_path": str(predictions_path.resolve()),
        "qa_path": str(qa_path.resolve()),
        "config_path": str(args.config.resolve()),
        "dataset": args.dataset,
        "total_failed_examples": total,
        "usable_query_count": usable_queries,
        "usable_query_rate": (usable_queries / total if total else 0.0),
        "parsed_without_fallback_count": parsed_without_fallback,
        "parsed_without_fallback_rate": (parsed_without_fallback / total if total else 0.0),
        "fallback_used_count": fallback_used,
        "fallback_used_rate": (fallback_used / total if total else 0.0),
        "failure_reason_counts": dict(failure_reason_counter),
        "request_error_counts": dict(request_error_counter),
        "query_count_distribution": dict(Counter(result["query_count"] for result in replay_results)),
    }

    save_json(output_dir / "summary.json", summary)
    with (output_dir / "replay_results.jsonl").open("w", encoding="utf-8") as f:
        for row in replay_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
