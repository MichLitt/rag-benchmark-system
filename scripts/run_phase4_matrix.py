"""Batch runner for Phase 4 experiment matrix.

Iterates over 5 configs × 3 datasets, calling run_naive_rag_baseline.py
for each combination.

Usage:
    uv run python scripts/run_phase4_matrix.py --dry-run
    uv run python scripts/run_phase4_matrix.py --configs C1,C2 --datasets nq
    uv run python scripts/run_phase4_matrix.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Config name -> (yaml path, use_reranker, extra CLI args)
CONFIGS: dict[str, tuple[Path, bool, list[str]]] = {
    "C1_dense_only": (
        ROOT / "config/phase4/C1_dense_only.yaml",
        False,
        [],
    ),
    "C2_dense_rerank": (
        ROOT / "config/phase4/C2_dense_rerank.yaml",
        True,
        ["--reranker-retriever-rank-weight", "0.4"],
    ),
    "C3_dense_rerank_topk5": (
        ROOT / "config/phase4/C3_dense_rerank_topk5.yaml",
        True,
        ["--reranker-retriever-rank-weight", "0.4"],
    ),
    "C4_rerank_hyde": (
        ROOT / "config/phase4/C4_rerank_hyde.yaml",
        True,
        [
            "--reranker-retriever-rank-weight", "0.4",
            "--query-expansion-mode", "hyde",
            "--query-expansion-datasets", "hotpotqa,nq,triviaqa",
        ],
    ),
    "C5_rerank_decompose": (
        ROOT / "config/phase4/C5_rerank_decompose.yaml",
        True,
        [
            "--reranker-retriever-rank-weight", "0.4",
            "--query-expansion-mode", "auto",
            "--query-expansion-datasets", "nq",
        ],
    ),
}

DATASETS = ["hotpotqa", "nq", "triviaqa"]


def build_command(
    config_name: str,
    dataset: str,
    max_queries: int,
    output_root: Path,
) -> list[str]:
    yaml_path, use_reranker, extra_args = CONFIGS[config_name]
    output_dir = output_root / config_name

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_naive_rag_baseline.py"),
        "--config", str(yaml_path),
        "--datasets", dataset,
        "--max-queries", str(max_queries),
        "--output-root", str(output_dir),
        "--continue-on-generation-error",
        "--dedup-before-rerank",
    ]
    if use_reranker:
        cmd.append("--use-reranker")
    cmd.extend(extra_args)
    return cmd


def already_done(
    config_name: str,
    dataset: str,
    output_root: Path,
    min_queries: int = 50,
) -> bool:
    """Return True if a sufficiently large completed run exists for this combo."""
    config_dir = output_root / config_name
    if not config_dir.exists():
        return False
    import json
    for metrics_path in config_dir.rglob("metrics.json"):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            if (
                m.get("Dataset", "").lower() == dataset.lower()
                and int(m.get("NumQueries", 0)) >= min_queries
            ):
                return True
        except Exception:
            continue
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 experiment matrix.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (config, dataset) combos that already have a metrics.json.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(CONFIGS.keys()),
        help="Comma-separated config names to run (default: all).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated dataset names to run (default: all).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=200,
        help="Max queries per dataset (default: 200).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "runs" / "phase4_matrix",
        help="Output root directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for c in configs:
        if c not in CONFIGS:
            print(f"ERROR: Unknown config '{c}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)

    jobs: list[tuple[str, str, list[str]]] = []
    for config_name in configs:
        for dataset in datasets:
            cmd = build_command(config_name, dataset, args.max_queries, args.output_root)
            jobs.append((config_name, dataset, cmd))

    print(f"Phase 4 matrix: {len(configs)} configs × {len(datasets)} datasets = {len(jobs)} runs")
    print(f"Max queries per dataset: {args.max_queries}")
    print(f"Output root: {args.output_root}")
    print()

    if args.dry_run:
        for config_name, dataset, cmd in jobs:
            print(f"[{config_name} / {dataset}]")
            print(f"  {' '.join(cmd)}")
            print()
        print("(dry run — nothing executed)")
        return

    total = len(jobs)
    completed = 0
    failed: list[tuple[str, str, int]] = []
    matrix_start = time.time()

    for config_name, dataset, cmd in jobs:
        completed += 1
        print(f"\n{'='*60}")
        print(f"[{completed}/{total}] {config_name} / {dataset}")
        print(f"{'='*60}")

        if args.skip_existing and already_done(config_name, dataset, args.output_root):
            print(f"  SKIP (already completed)")
            continue

        run_start = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))

        elapsed = time.time() - run_start
        if result.returncode != 0:
            failed.append((config_name, dataset, result.returncode))
            print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        else:
            print(f"  OK ({elapsed:.0f}s)")

    total_elapsed = time.time() - matrix_start
    print(f"\n{'='*60}")
    print(f"Matrix complete: {completed - len(failed)}/{total} succeeded in {total_elapsed:.0f}s")
    if failed:
        print(f"Failed runs ({len(failed)}):")
        for config_name, dataset, rc in failed:
            print(f"  {config_name} / {dataset} (exit code {rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
