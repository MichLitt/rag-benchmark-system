from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable, Iterator

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus import iter_corpus_documents, load_documents
from src.retrieval.docstore import save_docstore
from src.retrieval.tokenize import simple_tokenize
from src.types import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes from local corpus.")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/raw/corpus/wiki_passages/passages.jsonl.gz"),
        help="Input corpus JSONL/JSONL.GZ path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/indexes/wiki_passages"),
        help="Output index directory.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=100_000,
        help="Maximum docs to index. Use <=0 to index all docs from source.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for dense embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Dense embedding batch size.")
    parser.add_argument(
        "--faiss-index-type",
        type=str,
        choices=["flat", "ivf"],
        default="flat",
        help="FAISS index type.",
    )
    parser.add_argument("--ivf-nlist", type=int, default=1024, help="FAISS IVF nlist.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=50_000,
        help="Training sample size for IVF. Ignored for flat.",
    )
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 index build.")
    parser.add_argument("--skip-dense", action="store_true", help="Skip FAISS index build.")
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _combine_text(doc: Document) -> str:
    if doc.title and doc.title != doc.text:
        return f"{doc.title}. {doc.text}"
    return doc.text


def _iter_text_batches(texts: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def _iter_doc_batches(
    corpus_path: Path, max_docs: int | None, batch_size: int
) -> Iterator[list[Document]]:
    batch: list[Document] = []
    count = 0
    for doc in iter_corpus_documents(corpus_path):
        batch.append(doc)
        count += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
        if max_docs is not None and count >= max_docs:
            break
    if batch:
        yield batch


def _build_bm25(docs: list[Document], out_path: Path) -> None:
    tokenized = [simple_tokenize(_combine_text(doc)) for doc in tqdm(docs, desc="Tokenizing for BM25")]
    bm25 = BM25Okapi(tokenized)
    with out_path.open("wb") as f:
        pickle.dump(bm25, f)


def _make_ivf_index(dim: int, nlist: int) -> faiss.IndexIVFFlat:
    quantizer = faiss.IndexFlatIP(dim)
    return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)


def _build_dense_streaming(
    corpus_path: Path,
    max_docs: int | None,
    out_index_path: Path,
    out_cfg_path: Path,
    docstore_path: Path,
    model_name: str,
    batch_size: int,
    index_type: str,
    ivf_nlist: int,
    train_size: int,
) -> int:
    """Stream corpus, write docstore and FAISS index without loading all docs into RAM."""
    model = SentenceTransformer(model_name)

    # Determine embedding dimension from a single sample
    sample = next(iter_corpus_documents(corpus_path))
    first_vec = model.encode([_combine_text(sample)], convert_to_numpy=True, normalize_embeddings=True)
    dim = int(first_vec.shape[1])

    if index_type == "ivf":
        # IVF needs training data first — sample train_size docs into RAM
        train_n = max(ivf_nlist * 10, train_size)
        train_texts: list[str] = []
        for doc in iter_corpus_documents(corpus_path):
            train_texts.append(_combine_text(doc))
            if len(train_texts) >= train_n:
                break
        train_emb = model.encode(
            train_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        index: faiss.Index = _make_ivf_index(dim, ivf_nlist)
        print(f"Training IVF index on {len(train_emb)} samples...")
        index.train(train_emb)
        del train_texts, train_emb
    else:
        index = faiss.IndexFlatIP(dim)

    # Stream corpus: write docstore line-by-line, encode + add to FAISS batch-by-batch
    total = 0
    docstore_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(docstore_path, "wt", encoding="utf-8") if str(docstore_path).endswith(".gz") \
            else docstore_path.open("w", encoding="utf-8") as ds_file:
        for batch in tqdm(_iter_doc_batches(corpus_path, max_docs, batch_size), desc="Encoding + indexing"):
            # Write docstore entries
            for doc in batch:
                row = {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}
                ds_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Encode and add to FAISS
            texts = [_combine_text(doc) for doc in batch]
            emb = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32, copy=False)
            index.add(emb)
            total += len(batch)
            if total % 500_000 == 0:
                print(f"  {total:,} docs indexed...")

    faiss.write_index(index, str(out_index_path))
    dense_cfg = {
        "embedding_model": model_name,
        "index_type": index_type,
        "dimension": dim,
        "num_docs": total,
    }
    with out_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(dense_cfg, f, ensure_ascii=False, indent=2)
    return total


def _build_dense(
    docs: list[Document],
    out_index_path: Path,
    out_cfg_path: Path,
    model_name: str,
    batch_size: int,
    index_type: str,
    ivf_nlist: int,
    train_size: int,
) -> None:
    model = SentenceTransformer(model_name)
    texts = [_combine_text(doc) for doc in docs]
    if not texts:
        raise ValueError("No documents provided for dense index build.")

    first_vec = model.encode([texts[0]], convert_to_numpy=True, normalize_embeddings=True)
    dim = int(first_vec.shape[1])

    if index_type == "flat":
        index: faiss.Index = faiss.IndexFlatIP(dim)
    else:
        index = _make_ivf_index(dim, ivf_nlist)
        train_n = min(len(texts), max(ivf_nlist * 10, train_size))
        train_texts = texts[:train_n]
        train_emb = model.encode(
            train_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        print(f"Training IVF index on {len(train_emb)} samples...")
        index.train(train_emb)

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for batch_texts in tqdm(_iter_text_batches(texts, batch_size), total=total_batches, desc="Encoding + indexing"):
        emb = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        index.add(emb)

    faiss.write_index(index, str(out_index_path))
    dense_cfg = {
        "embedding_model": model_name,
        "index_type": index_type,
        "dimension": dim,
        "num_docs": len(docs),
    }
    with out_cfg_path.open("w", encoding="utf-8") as f:
        json.dump(dense_cfg, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    max_docs = args.max_docs if args.max_docs and args.max_docs > 0 else None

    manifest_path = args.output_dir / "manifest.json"
    artifacts: dict[str, str | int] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
            if isinstance(existing, dict):
                artifacts.update(existing)

    artifacts["corpus_path"] = str(args.corpus_path)
    docstore_path = args.output_dir / "docstore.jsonl"

    if not args.skip_dense:
        # Use streaming path: docstore is written during the encoding pass
        faiss_index_path = args.output_dir / "faiss.index"
        dense_cfg_path = args.output_dir / "dense_config.json"
        print(f"Building dense FAISS index (streaming) from: {args.corpus_path}")
        num_docs = _build_dense_streaming(
            corpus_path=args.corpus_path,
            max_docs=max_docs,
            out_index_path=faiss_index_path,
            out_cfg_path=dense_cfg_path,
            docstore_path=docstore_path,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            index_type=args.faiss_index_type,
            ivf_nlist=args.ivf_nlist,
            train_size=args.train_size,
        )
        artifacts["docstore_path"] = str(docstore_path)
        artifacts["num_docs"] = num_docs
        artifacts["faiss_index_path"] = str(faiss_index_path)
        artifacts["dense_config_path"] = str(dense_cfg_path)
        print(f"Dense index done: {num_docs:,} docs")
    else:
        # Dense skipped: still need docstore for BM25
        print(f"Loading corpus from: {args.corpus_path}")
        docs = load_documents(args.corpus_path, max_docs=max_docs)
        print(f"Loaded {len(docs)} documents")
        save_docstore(docstore_path, docs)
        artifacts["docstore_path"] = str(docstore_path)
        artifacts["num_docs"] = len(docs)

        if not args.skip_bm25:
            bm25_path = args.output_dir / "bm25.pkl"
            print("Building BM25 index...")
            _build_bm25(docs, bm25_path)
            artifacts["bm25_path"] = str(bm25_path)

    if not args.skip_bm25 and not args.skip_dense:
        # BM25 still requires all docs in RAM; load from already-written docstore
        from src.retrieval.docstore import load_docstore
        print("Loading docstore for BM25 build...")
        docs_for_bm25 = load_docstore(docstore_path)
        bm25_path = args.output_dir / "bm25.pkl"
        print("Building BM25 index...")
        _build_bm25(docs_for_bm25, bm25_path)
        artifacts["bm25_path"] = str(bm25_path)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=2)
    print(f"Done. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
