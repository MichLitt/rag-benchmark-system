"""CLI: parse a PDF and write chunks to a JSONL docstore.

Usage
-----
    uv run python scripts/ingest_documents.py \\
        --input path/to/document.pdf \\
        --output data/indexes/my_index/docstore.jsonl \\
        --title "My Document" \\
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

from src.ingestion.chunker import TokenAwareChunker, make_doc_id_prefix
from src.ingestion.factory import get_parser
from src.logging_utils import configure_logging, get_logger
from src.retrieval.docstore import save_docstore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest a PDF document into a JSONL docstore.",
    )
    p.add_argument("--input", required=True, type=Path, help="Input PDF file path.")
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output JSONL docstore path (parent dirs created automatically).",
    )
    p.add_argument("--title", default="", help="Document title (defaults to filename stem).")
    p.add_argument("--parser", default="pdf", choices=["pdf"], help="Parser mode.")
    p.add_argument("--chunk-size", type=int, default=256, help="Max tokens per chunk.")
    p.add_argument("--overlap", type=int, default=32, help="Token overlap between chunks.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    input_path: Path = args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    title = args.title or input_path.stem
    source = input_path.name

    # Parse
    parser = get_parser(args.parser)
    pages = parser.parse(input_path)
    if not pages:
        logger.error("No text could be extracted from '%s' — aborting.", input_path)
        sys.exit(1)

    # Chunk
    chunker = TokenAwareChunker(chunk_size=args.chunk_size, overlap=args.overlap)
    prefix = make_doc_id_prefix(source)
    chunks = chunker.chunk(pages, doc_id_prefix=prefix, title=title, source=source)

    if not chunks:
        logger.error("Chunker produced zero chunks — aborting.")
        sys.exit(1)

    total_tokens = sum(
        len(chunker._enc.encode(c.text)) for c in chunks  # noqa: SLF001
    )
    logger.info(
        "Ingestion complete: %d chunks, %d total tokens → %s",
        len(chunks), total_tokens, args.output,
    )

    # Persist
    save_docstore(args.output, chunks)
    logger.info("Docstore written to '%s'", args.output)


if __name__ == "__main__":
    main()
