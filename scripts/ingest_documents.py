#!/usr/bin/env python3
"""Ingest PDF documents into a JSONL docstore.

Usage:
    uv run python scripts/ingest_documents.py \\
        --input doc1.pdf doc2.pdf \\
        --output data/indexes/my_index/docstore.jsonl \\
        --chunk-size 256 \\
        --overlap 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import TokenAwareChunker
from src.ingestion.factory import get_parser
from src.retrieval.docstore import save_docstore
from src.types import Document


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse PDF files into a chunked JSONL docstore.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", type=Path, required=True, nargs="+",
        metavar="PDF",
        help="One or more PDF files to ingest.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        metavar="JSONL",
        help="Output JSONL docstore path.",
    )
    p.add_argument("--chunk-size", type=int, default=256, help="Max tokens per chunk.")
    p.add_argument("--overlap", type=int, default=32, help="Token overlap between consecutive chunks.")
    p.add_argument("--mode", type=str, default="pdf", choices=["pdf"], help="Parser mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    parser = get_parser(args.mode)
    chunker = TokenAwareChunker(chunk_size=args.chunk_size, overlap=args.overlap)

    all_docs: list[Document] = []
    for pdf_path in args.input:
        if not pdf_path.exists():
            print(f"[ERROR] File not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)
        pages = parser.parse(pdf_path)
        page_count = len(pages)
        chunks: list[Document] = []
        for page_doc in pages:
            chunks.extend(chunker.chunk(page_doc))
        print(
            f"  {pdf_path.name}: {page_count} pages → {len(chunks)} chunks"
        )
        all_docs.extend(chunks)

    save_docstore(args.output, all_docs)
    print(f"\nWrote {len(all_docs)} chunks to {args.output}")


if __name__ == "__main__":
    main()
