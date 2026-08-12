"""Build a deterministic BM25 index from a versioned text corpus manifest.

This is intentionally separate from the runtime PDF ingestion queue. It is for
small, checked-in engineering documentation that must be rebuilt identically for
controlled evaluations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi

from src.ingestion.chunker import TokenAwareChunker, make_doc_id_prefix
from src.ingestion.pdf_parser import PageSpan
from src.retrieval.docstore import save_docstore
from src.retrieval.tokenize import simple_tokenize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("documents"), list):
        raise ValueError("manifest must be a JSON object with a documents list")
    return value


def build_text_index(
    *,
    manifest_path: Path,
    workspace_root: Path,
    output_dir: Path,
    chunk_size: int = 256,
    overlap: int = 32,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    workspace_root = workspace_root.resolve()
    output_dir = output_dir.resolve()
    manifest = load_manifest(manifest_path)
    index_name = manifest.get("index_name")
    if not isinstance(index_name, str) or not index_name:
        raise ValueError("manifest index_name must be a non-empty string")

    chunker = TokenAwareChunker(chunk_size=chunk_size, overlap=overlap)
    documents = []
    sources: list[dict[str, str]] = []
    for relative in manifest["documents"]:
        if not isinstance(relative, str) or not relative:
            raise ValueError("manifest documents must contain non-empty strings")
        path = (workspace_root / relative).resolve()
        if workspace_root not in path.parents or not path.is_file():
            raise ValueError(f"manifest document is missing or outside workspace: {relative}")
        text = path.read_text(encoding="utf-8")
        chunks = chunker.chunk(
            [PageSpan(page_num=1, text=text)],
            doc_id_prefix=make_doc_id_prefix(relative),
            title=path.stem,
            source=relative,
        )
        if not chunks:
            raise ValueError(f"manifest document produced no chunks: {relative}")
        documents.extend(chunks)
        sources.append({"path": relative, "sha256": _sha256(path)})

    output_dir.mkdir(parents=True, exist_ok=True)
    docstore_path = output_dir / "docstore.jsonl"
    save_docstore(docstore_path, documents)
    tokenized = [simple_tokenize(f"{doc.title} {doc.text}".strip()) for doc in documents]
    with (output_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(BM25Okapi(tokenized), handle)

    build = {
        "schema_version": "text-index-build/v1",
        "index_name": index_name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "sources": sources,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "doc_count": len(documents),
        "docstore_sha256": _sha256(docstore_path),
    }
    (output_dir / "corpus-build-manifest.json").write_text(
        json.dumps(build, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    result = build_text_index(
        manifest_path=args.manifest,
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
