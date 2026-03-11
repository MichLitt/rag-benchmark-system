from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Wikipedia passages corpus for retrieval indexing.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nicCli/Wikipedia_Passages_1M",
        help="HF dataset id containing Wikipedia passages.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to stream from the corpus dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/corpus/wiki_passages"),
        help="Directory to save corpus files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1_000_000,
        help="Maximum rows to save. Use <=0 for full split.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _split_contents(contents: str) -> tuple[str, str]:
    if "[SEP]" not in contents:
        raw = contents.strip()
        if "\n" in raw:
            maybe_title, maybe_text = raw.split("\n", 1)
            title = maybe_title.strip().strip('"').strip()
            text = maybe_text.strip()
            if title and text:
                return title, text
        return "", raw
    title, text = contents.split("[SEP]", 1)
    return title.strip(), text.strip()


def _iter_corpus(dataset: str, split: str) -> Iterable[dict]:
    ds = load_dataset(dataset, split=split, streaming=True)
    for row in ds:
        yield row


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    out_path = args.output_dir / "passages.jsonl.gz"
    count = 0
    max_rows = args.max_rows

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for row in _iter_corpus(args.dataset, args.split):
            title, text = _split_contents(str(row.get("contents", "")))
            normalized = {
                "doc_id": str(row.get("id", "")),
                "title": title,
                "text": text,
            }
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
            if count % 100_000 == 0:
                print(f"Saved {count} passages...")
            if max_rows > 0 and count >= max_rows:
                break

    manifest = {
        "source_dataset": args.dataset,
        "split": args.split,
        "rows": count,
        "output_file": str(out_path),
        "max_rows": max_rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {count} rows to {out_path}")


if __name__ == "__main__":
    main()
