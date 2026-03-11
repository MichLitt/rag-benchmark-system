from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus import iter_corpus_documents
from src.retrieval.docstore import save_docstore
from src.retrieval.postprocess import normalize_title
from src.retrieval.tokenize import simple_tokenize
from src.types import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a title-only BM25 index from a passage corpus.")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/raw/corpus/wiki18_21m/passages.jsonl.gz"),
        help="Input corpus JSONL/JSONL.GZ path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes/wiki18_21m_title_bm25"),
        help="Output directory for title-only BM25 artifacts.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Only read the first N source docs. Use <=0 to process the full corpus.",
    )
    parser.add_argument(
        "--representative-docids-per-title",
        type=int,
        default=3,
        help="How many representative chunk doc_ids to keep per unique title in metadata.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    max_docs = args.max_docs if args.max_docs and args.max_docs > 0 else None
    keep_docids = int(args.representative_docids_per_title)
    if keep_docids <= 0:
        raise ValueError(
            f"representative-docids-per-title must be > 0, got {args.representative_docids_per_title}"
        )

    aggregated: OrderedDict[str, dict[str, object]] = OrderedDict()
    total_docs = 0
    for doc in tqdm(iter_corpus_documents(args.corpus_path), desc="Scanning corpus for unique titles"):
        total_docs += 1
        if max_docs is not None and total_docs > max_docs:
            break
        normalized = normalize_title(doc.title)
        if not normalized:
            continue
        entry = aggregated.get(normalized)
        if entry is None:
            entry = {
                "normalized_title": normalized,
                "display_title": doc.title,
                "chunk_count": 0,
                "representative_doc_ids": [],
            }
            aggregated[normalized] = entry
        entry["chunk_count"] = int(entry["chunk_count"]) + 1
        representative_doc_ids = entry["representative_doc_ids"]
        if (
            isinstance(representative_doc_ids, list)
            and doc.doc_id
            and len(representative_doc_ids) < keep_docids
        ):
            representative_doc_ids.append(doc.doc_id)

    title_docs = [
        Document(
            doc_id=str(entry["normalized_title"]),
            title=str(entry["display_title"]),
            text=str(entry["display_title"]),
        )
        for entry in aggregated.values()
    ]
    tokenized_titles = [simple_tokenize(doc.title or doc.text) for doc in tqdm(title_docs, desc="Tokenizing titles")]
    bm25 = BM25Okapi(tokenized_titles)

    docstore_path = args.output_dir / "docstore.jsonl"
    bm25_path = args.output_dir / "bm25.pkl"
    metadata_path = args.output_dir / "title_metadata.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    save_docstore(docstore_path, title_docs)
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    with metadata_path.open("w", encoding="utf-8") as f:
        for entry in aggregated.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    manifest = {
        "index_format": "title_bm25",
        "corpus_path": str(args.corpus_path.resolve()),
        "docstore_path": str(docstore_path.resolve()),
        "bm25_path": str(bm25_path.resolve()),
        "title_metadata_path": str(metadata_path.resolve()),
        "normalization_mode": "src.retrieval.postprocess.normalize_title",
        "total_unique_titles": len(title_docs),
        "source_docs_seen": total_docs if max_docs is None else min(total_docs, max_docs),
        "representative_docids_per_title": keep_docids,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path.resolve()),
                "total_unique_titles": len(title_docs),
                "source_docs_seen": manifest["source_docs_seen"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
