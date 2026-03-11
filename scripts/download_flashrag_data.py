from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


HF_REPO = "RUC-NLPIR/FlashRAG_datasets"


@dataclass(frozen=True)
class DatasetSpec:
    config: str
    eval_splits: list[str]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec(config="hotpotqa", eval_splits=["dev"]),
    "nq": DatasetSpec(config="nq", eval_splits=["test"]),
    "triviaqa": DatasetSpec(config="triviaqa", eval_splits=["test"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and persist FlashRAG benchmark datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/flashrag"),
        help="Directory to save datasets.",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="If set, save all splits instead of only eval split(s).",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_hotpot_titles(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata") or {}
    supporting = metadata.get("supporting_facts") or {}
    titles = supporting.get("title") or []
    seen: set[str] = set()
    deduped: list[str] = []
    for title in titles:
        if title not in seen:
            seen.add(title)
            deduped.append(title)
    return deduped


def _normalize_row(dataset_name: str, row: dict[str, Any]) -> dict[str, Any]:
    answers = row.get("golden_answers") or []
    normalized = {
        "id": str(row.get("id", "")),
        "question": row.get("question", ""),
        "golden_answers": [str(a) for a in answers],
        "gold_answer": str(answers[0]) if answers else "",
    }
    if dataset_name == "hotpotqa":
        normalized["gold_titles"] = _extract_hotpot_titles(row)
    return normalized


def _write_jsonl(dataset_name: str, split_name: str, ds: Dataset, out_path: Path) -> dict[str, Any]:
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            normalized = _normalize_row(dataset_name, row)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    return {"rows": count, "path": str(out_path)}


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    _ensure_dir(output_dir)

    manifest: dict[str, Any] = {"source_repo": HF_REPO, "datasets": {}}

    for dataset_name, spec in DATASET_SPECS.items():
        print(f"\n=== Downloading {dataset_name} ({spec.config}) ===")
        ds_dict = load_dataset(HF_REPO, spec.config)
        splits = list(ds_dict.keys()) if args.all_splits else spec.eval_splits
        manifest["datasets"][dataset_name] = {}

        for split in splits:
            if split not in ds_dict:
                print(f"Skip split {split} (not found)")
                continue

            split_ds = ds_dict[split]
            split_dir = output_dir / dataset_name / split
            _ensure_dir(split_dir)

            hf_disk_path = split_dir / "hf_dataset"
            split_ds.save_to_disk(str(hf_disk_path))

            jsonl_path = split_dir / "qa.jsonl"
            stats = _write_jsonl(dataset_name, split, split_ds, jsonl_path)
            stats["hf_dataset_path"] = str(hf_disk_path)

            manifest["datasets"][dataset_name][split] = stats
            print(f"Saved {dataset_name}/{split}: {stats['rows']} rows")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
