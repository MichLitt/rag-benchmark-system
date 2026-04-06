"""CLI: build a SPLADE inverted index from a JSONL docstore.

Encodes every document in *docstore_path* using the SPLADE MLM model and saves
the resulting sparse matrix as ``splade_index.npz`` alongside a
``splade_config.json`` in the same directory.

Usage
-----
    uv run python scripts/build_splade_index.py \\
        --docstore  data/indexes/my_index/docstore.jsonl \\
        --output    data/indexes/my_index \\
        --model     naver-splade/splade-cocondenser-ensembledistil \\
        --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.logging_utils import configure_logging, get_logger
from src.retrieval.docstore import load_docstore
from src.retrieval.splade import HFSpladeEncoder, SPLADE_MODEL_NAME, build_splade_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a SPLADE sparse index from a JSONL docstore.")
    p.add_argument(
        "--docstore", required=True, type=Path,
        help="Input JSONL docstore path.",
    )
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output directory (splade_index.npz + splade_config.json written here).",
    )
    p.add_argument(
        "--model", default=SPLADE_MODEL_NAME,
        help="HuggingFace SPLADE model name.",
    )
    p.add_argument("--device", default=None, help="Device: 'cpu' or 'cuda' (auto-detected).")
    p.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    p.add_argument("--log-level", default="INFO", help="Logging level.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    docstore_path: Path = args.docstore
    if not docstore_path.exists():
        logger.error("Docstore not found: %s", docstore_path)
        sys.exit(1)

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading docstore from '%s' …", docstore_path)
    docs = load_docstore(docstore_path)
    logger.info("Loaded %d documents.", len(docs))

    encoder = HFSpladeEncoder(model_name=args.model, device=args.device)
    matrix = build_splade_index(docs, encoder, batch_size=args.batch_size)

    # Save index matrix
    import scipy.sparse
    index_path = out_dir / "splade_index.npz"
    scipy.sparse.save_npz(str(index_path), matrix)
    logger.info("Saved SPLADE index to '%s' (shape=%s, nnz=%d)", index_path, matrix.shape, matrix.nnz)

    # Save config
    cfg = {
        "model_name": args.model,
        "vocab_size": encoder.vocab_size,
        "num_docs": len(docs),
    }
    config_path = out_dir / "splade_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Saved config to '%s'", config_path)


if __name__ == "__main__":
    main()
