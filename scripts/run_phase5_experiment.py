"""Batch runner for Phase 5 ablation experiment.

Tests three generation improvements against a GLM-5.x baseline, isolating
each improvement's contribution with a 5-config × 3-dataset matrix:

    C1  GLM baseline       — fixed timeout/budget, no other changes
    C2  + answer postproc  — strip hedging prefixes before EM/F1
    C3  + citation+NLI     — citation-constrained generation with HHEM scoring
    C4  + dataset prompts  — dataset-specific system prompts
    C5  full stack         — all three improvements combined

Usage:
    uv run python scripts/run_phase5_experiment.py --dry-run
    uv run python scripts/run_phase5_experiment.py --configs C1,C2 --datasets nq
    uv run python scripts/run_phase5_experiment.py --max-queries 200
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Config name → (yaml path, postprocess_answers, needs_dataset_prompt, use_citation)
# needs_dataset_prompt=True means --generation-dataset is injected per dataset run.
CONFIGS: dict[str, tuple[Path, bool, bool, bool]] = {
    "C1_glm_baseline": (
        ROOT / "config/phase5/C1_glm_baseline.yaml",
        False,  # postprocess_answers
        False,  # needs_dataset_prompt
        False,  # use_citation (C3/C5 only)
    ),
    "C2_glm_postprocess": (
        ROOT / "config/phase5/C2_glm_postprocess.yaml",
        True,
        False,
        False,
    ),
    "C3_glm_citation": (
        ROOT / "config/phase5/C3_glm_citation.yaml",
        True,
        False,
        True,
    ),
    "C4_glm_dataset_prompts": (
        ROOT / "config/phase5/C4_glm_dataset_prompts.yaml",
        True,
        True,   # inject --generation-dataset per run
        False,
    ),
    "C5_glm_full": (
        ROOT / "config/phase5/C5_glm_full.yaml",
        True,
        True,
        True,
    ),
}

DATASETS = ["hotpotqa", "nq", "triviaqa"]


def build_command(
    config_name: str,
    dataset: str,
    max_queries: int,
    output_root: Path,
) -> list[str]:
    yaml_path, postprocess, needs_dataset_prompt, _ = CONFIGS[config_name]
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
        "--use-reranker",
        "--reranker-retriever-rank-weight", "0.4",
    ]
    if postprocess:
        cmd.append("--postprocess-answers")
    if needs_dataset_prompt:
        cmd.extend(["--generation-dataset", dataset])
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
    parser = argparse.ArgumentParser(description="Run Phase 5 generation ablation experiment.")
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
        help="Max queries per dataset per config (default: 200).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments" / "runs" / "phase5_ablation",
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

    print(f"Phase 5 ablation: {len(configs)} configs × {len(datasets)} datasets = {len(jobs)} runs")
    print(f"Max queries per run: {args.max_queries}")
    print(f"Output root: {args.output_root}")
    print()
    print("Ablation design:")
    print("  C1 GLM baseline       — fixed timeout/budget, isolates true GLM capability")
    print("  C2 + postprocess      — zero-cost EM/F1 gain from hedge stripping")
    print("  C3 + citation+NLI     — citation-constrained generation + HHEM scoring (slow: loads DeBERTa)")
    print("  C4 + dataset prompts  — dataset-specific answer format constraints")
    print("  C5 full stack         — all three combined, compound effect")
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
        print(f"\n{'='*70}")
        print(f"[{completed}/{total}] {config_name} / {dataset}")
        print(f"{'='*70}")

        if args.skip_existing and already_done(config_name, dataset, args.output_root):
            print("  SKIP (already completed)")
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
    print(f"\n{'='*70}")
    print(f"Phase 5 complete: {completed - len(failed)}/{total} succeeded in {total_elapsed:.0f}s")

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for config_name, dataset, rc in failed:
            print(f"  {config_name} / {dataset} (exit code {rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
