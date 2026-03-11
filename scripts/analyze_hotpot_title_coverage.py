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
    parser = argparse.ArgumentParser(description="Analyze HotpotQA gold-title coverage in a corpus.")
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=Path("data/raw/flashrag/hotpotqa/dev/qa.jsonl"),
        help="Path to normalized HotpotQA qa.jsonl.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/raw/corpus/wiki_passages/passages.jsonl.gz"),
        help="Path to corpus jsonl/jsonl.gz.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Only scan first N docs from corpus; <=0 means full file.",
    )
    return parser.parse_args()


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _load_title_sets(corpus_path: Path, max_docs: int) -> tuple[int, set[str], set[str]]:
    exact: set[str] = set()
    loose: set[str] = set()
    rows = 0
    for doc in iter_corpus_documents(corpus_path):
        rows += 1
        if doc.title:
            exact.add(doc.title.lower())
            loose.add(_normalize(doc.title))
        if max_docs > 0 and rows >= max_docs:
            break
    return rows, exact, loose


def _safe_pct(hit: int, total: int) -> float:
    return (hit / total * 100.0) if total else 0.0


def main() -> None:
    args = parse_args()
    rows, title_exact, title_loose = _load_title_sets(args.corpus_path, int(args.max_docs))

    query_total = 0
    mention_total = 0
    mention_hit_exact = 0
    mention_hit_loose = 0
    query_any_exact = 0
    query_any_loose = 0
    query_all_exact = 0
    query_all_loose = 0

    with args.qa_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gold_titles = [str(t) for t in row.get("gold_titles", []) if str(t).strip()]
            if not gold_titles:
                continue

            query_total += 1
            exact_titles = [t.lower() for t in gold_titles]
            loose_titles = [_normalize(t) for t in gold_titles]

            mention_total += len(exact_titles)
            hit_exact = [t for t in exact_titles if t in title_exact]
            hit_loose = [t for t in loose_titles if t in title_loose]
            mention_hit_exact += len(hit_exact)
            mention_hit_loose += len(hit_loose)

            if hit_exact:
                query_any_exact += 1
            if hit_loose:
                query_any_loose += 1
            if len(hit_exact) == len(exact_titles):
                query_all_exact += 1
            if len(hit_loose) == len(loose_titles):
                query_all_loose += 1

    report = {
        "qa_path": str(args.qa_path),
        "corpus_path": str(args.corpus_path),
        "rows_scanned": rows,
        "unique_titles_exact": len(title_exact),
        "unique_titles_loose": len(title_loose),
        "queries": query_total,
        "title_mentions": mention_total,
        "mention_exact_coverage_pct": round(_safe_pct(mention_hit_exact, mention_total), 2),
        "mention_loose_coverage_pct": round(_safe_pct(mention_hit_loose, mention_total), 2),
        "query_any_exact_pct": round(_safe_pct(query_any_exact, query_total), 2),
        "query_any_loose_pct": round(_safe_pct(query_any_loose, query_total), 2),
        "query_all_exact_pct": round(_safe_pct(query_all_exact, query_total), 2),
        "query_all_loose_pct": round(_safe_pct(query_all_loose, query_total), 2),
        "query_any_exact": query_any_exact,
        "query_all_exact": query_all_exact,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
