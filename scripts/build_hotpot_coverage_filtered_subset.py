from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus import iter_corpus_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a HotpotQA subset filtered by corpus title coverage.")
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=Path("data/raw/flashrag/hotpotqa/dev/qa.jsonl"),
        help="Input HotpotQA qa.jsonl path.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/raw/corpus/wiki18_21m/passages.jsonl.gz"),
        help="Corpus JSONL/JSONL.GZ path used for title coverage.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/filtered/hotpotqa_all_gold_covered.jsonl"),
        help="Output subset path.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def main() -> None:
    args = parse_args()

    corpus_titles: set[str] = set()
    for doc in iter_corpus_documents(args.corpus_path):
        if doc.title:
            corpus_titles.add(_normalize(doc.title))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    with args.qa_path.open("r", encoding="utf-8") as src, args.output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            gold_titles = [_normalize(str(title)) for title in row.get("gold_titles", []) if str(title).strip()]
            if gold_titles and all(title in corpus_titles for title in gold_titles):
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    print(
        json.dumps(
            {
                "qa_path": str(args.qa_path),
                "corpus_path": str(args.corpus_path),
                "output_path": str(args.output_path),
                "total_queries": total,
                "kept_queries": kept,
                "kept_ratio": (kept / total if total else 0.0),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
