from __future__ import annotations

import argparse
import json
import shutil
import sys
from os import cpu_count
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus import iter_corpus_documents
from src.types import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sharded dense FAISS index from a corpus.")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/raw/corpus/wiki18_21m/passages.jsonl.gz"),
        help="Input corpus JSONL/JSONL.GZ path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"E:\rag-benchmark-indexes\wiki18_21m_dense_sharded"),
        help="Output root directory for the sharded index.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
        help="Maximum documents per shard.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer embedding model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="SentenceTransformer device, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["flat", "ivf_flat"],
        default="ivf_flat",
        help="FAISS index type written to each shard.",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=4096,
        help="Target IVF nlist when --index-type=ivf_flat.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=0,
        help="Training sample size per shard for IVF. Use 0 for max(200000, ivf_nlist*50).",
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=16,
        help="Default nprobe written to IVF manifests.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Only index the first N documents. Use <=0 to index the full corpus.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed shards under --output-root and remove any incomplete trailing shards.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _combine_text(doc: Document) -> str:
    if doc.title and doc.title != doc.text:
        return f"{doc.title}. {doc.text}"
    return doc.text


def _iter_doc_batches(
    corpus_path: Path,
    max_docs: int | None,
    batch_size: int,
    skip_docs: int = 0,
):
    batch: list[Document] = []
    total = 0
    for doc in iter_corpus_documents(corpus_path):
        total += 1
        if total <= skip_docs:
            if max_docs is not None and total >= max_docs:
                break
            continue
        batch.append(doc)
        if len(batch) >= batch_size:
            yield batch
            batch = []
        if max_docs is not None and total >= max_docs:
            break
    if batch:
        yield batch


def _write_doc_row(doc_file, offsets_file, doc: Document) -> None:
    raw = (
        json.dumps(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "page_start": doc.page_start,
                "page_end": doc.page_end,
                "section": doc.section,
                "source": doc.source,
                "extra_metadata": doc.extra_metadata,
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    offsets_file.write(int(doc_file.tell()).to_bytes(8, byteorder="little", signed=False))
    doc_file.write(raw)


def _make_flat_index(dim: int) -> faiss.Index:
    return faiss.IndexFlatIP(dim)


def _effective_train_size(index_type: str, ivf_nlist: int, train_size: int) -> int:
    if index_type != "ivf_flat":
        return 0
    if train_size > 0:
        return train_size
    return max(200_000, ivf_nlist * 50)


def _effective_nlist(target_nlist: int, train_vectors: int) -> int:
    if train_vectors <= 0:
        raise ValueError("train_vectors must be > 0")
    if train_vectors < 50:
        return 1
    return max(1, min(int(target_nlist), train_vectors // 50))


def _make_ivf_flat_index(dim: int, nlist: int) -> faiss.IndexIVFFlat:
    quantizer = faiss.IndexFlatIP(dim)
    return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)


def _parse_shard_id(shard_dir: Path) -> int | None:
    if not shard_dir.is_dir():
        return None
    try:
        prefix, suffix = shard_dir.name.split("_", maxsplit=1)
    except ValueError:
        return None
    if prefix != "shard" or not suffix.isdigit():
        return None
    return int(suffix)


def _load_completed_shard(
    shard_dir: Path,
    *,
    shard_id: int,
    doc_start: int,
    dimension: int,
    default_index_type: str,
    default_nprobe: int | None,
) -> tuple[dict[str, object], int] | None:
    faiss_index_path = shard_dir / "faiss.index"
    docstore_path = shard_dir / "docstore.jsonl"
    offsets_path = shard_dir / "docstore_offsets.bin"
    dense_cfg_path = shard_dir / "dense_config.json"
    required_paths = [faiss_index_path, docstore_path, offsets_path, dense_cfg_path]
    if not all(path.exists() and path.stat().st_size > 0 for path in required_paths):
        return None

    with dense_cfg_path.open("r", encoding="utf-8") as f:
        dense_cfg = json.load(f)

    num_docs = int(dense_cfg.get("num_docs") or 0)
    if num_docs <= 0:
        return None
    if offsets_path.stat().st_size != num_docs * 8:
        return None

    shard_index_type = str(dense_cfg.get("index_type") or default_index_type).strip().lower()
    shard_dimension = int(dense_cfg.get("dimension") or dimension)
    if shard_dimension != dimension:
        raise ValueError(
            f"Existing shard {shard_dir} has dimension {shard_dimension}, expected {dimension}"
        )

    shard_nlist = dense_cfg.get("nlist")
    if shard_index_type == "ivf_flat":
        if shard_nlist is None or int(shard_nlist) <= 0:
            return None
        shard_nprobe = int(dense_cfg.get("nprobe") or default_nprobe or 0)
        if shard_nprobe <= 0:
            return None
    else:
        shard_nprobe = None

    shard_info = {
        "shard_id": shard_id,
        "index_type": shard_index_type,
        "faiss_index_path": str(faiss_index_path.resolve()),
        "docstore_path": str(docstore_path.resolve()),
        "docstore_offsets_path": str(offsets_path.resolve()),
        "dense_config_path": str(dense_cfg_path.resolve()),
        "dimension": dimension,
        "nlist": int(shard_nlist) if shard_nlist is not None else None,
        "nprobe": shard_nprobe,
        "doc_start": doc_start,
        "doc_end": doc_start + num_docs,
    }
    return shard_info, num_docs


def _prepare_resume_state(
    output_root: Path,
    *,
    dimension: int,
    index_type: str,
    nprobe: int | None,
) -> tuple[list[dict[str, object]], int, int]:
    shard_dirs = sorted(
        ((shard_dir, _parse_shard_id(shard_dir)) for shard_dir in output_root.iterdir()),
        key=lambda item: (-1 if item[1] is None else item[1]),
    )
    completed_shards: list[dict[str, object]] = []
    total_docs = 0
    next_shard_id = 0
    cleanup = False

    for shard_dir, shard_id in shard_dirs:
        if shard_id is None:
            continue
        if cleanup:
            shutil.rmtree(shard_dir)
            continue
        if shard_id != next_shard_id:
            raise ValueError(
                f"Cannot resume from {output_root}: expected shard_{next_shard_id:03d}, found {shard_dir.name}"
            )
        loaded = _load_completed_shard(
            shard_dir,
            shard_id=shard_id,
            doc_start=total_docs,
            dimension=dimension,
            default_index_type=index_type,
            default_nprobe=nprobe,
        )
        if loaded is None:
            shutil.rmtree(shard_dir)
            cleanup = True
            continue
        shard_info, num_docs = loaded
        completed_shards.append(shard_info)
        total_docs += num_docs
        next_shard_id += 1

    return completed_shards, total_docs, next_shard_id


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_root)

    max_docs = args.max_docs if args.max_docs and args.max_docs > 0 else None
    model = SentenceTransformer(args.embedding_model, device=args.device)

    sample_doc = next(iter_corpus_documents(args.corpus_path))
    sample_emb = model.encode(
        [_combine_text(sample_doc)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = int(sample_emb.shape[1])

    shard_size = int(args.shard_size)
    if shard_size <= 0:
        raise ValueError(f"shard-size must be > 0, got {shard_size}")

    index_type = str(args.index_type).strip().lower()
    if index_type == "ivf_flat":
        if args.ivf_nlist <= 0:
            raise ValueError(f"ivf-nlist must be > 0, got {args.ivf_nlist}")
        if args.nprobe <= 0:
            raise ValueError(f"nprobe must be > 0, got {args.nprobe}")
    effective_train_size = _effective_train_size(index_type, int(args.ivf_nlist), int(args.train_size))

    shards: list[dict[str, object]] = []
    total_docs = 0
    shard_doc_count = 0
    shard_index = -1
    shard_start = 0
    shard_dir: Path | None = None
    dense_cfg_path: Path | None = None
    doc_file = None
    offsets_file = None
    index: faiss.Index | None = None
    shard_nlist: int | None = None
    pending_emb_batches: list[np.ndarray] = []
    pending_emb_count = 0

    if args.resume:
        shards, total_docs, next_shard_id = _prepare_resume_state(
            args.output_root,
            dimension=dim,
            index_type=index_type,
            nprobe=int(args.nprobe) if index_type == "ivf_flat" else None,
        )
        shard_index = next_shard_id - 1
        if shards:
            print(
                f"Resuming from shard_{next_shard_id:03d}; "
                f"reusing {len(shards)} completed shards covering {total_docs:,} docs"
            )
        else:
            print("Resume requested but no completed shards were found; starting from scratch")
        if max_docs is not None and total_docs >= max_docs:
            print(
                f"Resume state already covers max_docs={max_docs:,}; "
                "writing manifest from completed shards only"
            )

    def _materialize_shard_index() -> None:
        nonlocal index, pending_emb_batches, pending_emb_count, shard_nlist
        if index is not None:
            return
        if not pending_emb_batches:
            return

        train_vectors = np.vstack(pending_emb_batches).astype(np.float32, copy=False)
        shard_nlist = _effective_nlist(int(args.ivf_nlist), int(train_vectors.shape[0]))
        index = _make_ivf_flat_index(dim, shard_nlist)
        print(
            f"Training shard {shard_index:03d} IVF index on "
            f"{train_vectors.shape[0]:,} vectors with nlist={shard_nlist}"
        )
        index.train(train_vectors)
        index.add(train_vectors)
        pending_emb_batches = []
        pending_emb_count = 0

    def _append_embeddings(embeddings: np.ndarray) -> None:
        nonlocal index, pending_emb_batches, pending_emb_count
        if index_type == "flat":
            if index is None:
                index = _make_flat_index(dim)
            index.add(embeddings)
            return

        if index is not None:
            index.add(embeddings)
            return

        pending_emb_batches.append(embeddings)
        pending_emb_count += int(embeddings.shape[0])
        if pending_emb_count >= effective_train_size:
            _materialize_shard_index()

    def finalize_current_shard() -> dict[str, object] | None:
        nonlocal index, doc_file, offsets_file, shard_doc_count, dense_cfg_path, shard_dir, shard_nlist
        if shard_dir is None or dense_cfg_path is None:
            return None
        if index_type == "ivf_flat" and index is None:
            _materialize_shard_index()
        if index is None:
            return None
        finalized_nlist = shard_nlist
        faiss.write_index(index, str(shard_dir / "faiss.index"))
        with dense_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_model": args.embedding_model,
                    "index_type": index_type,
                    "dimension": dim,
                    "num_docs": shard_doc_count,
                    "nlist": finalized_nlist,
                    "nprobe": int(args.nprobe) if index_type == "ivf_flat" else None,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        shard_info = {
            "shard_id": shard_index,
            "index_type": index_type,
            "faiss_index_path": str((shard_dir / "faiss.index").resolve()),
            "docstore_path": str((shard_dir / "docstore.jsonl").resolve()),
            "docstore_offsets_path": str((shard_dir / "docstore_offsets.bin").resolve()),
            "dense_config_path": str((shard_dir / "dense_config.json").resolve()),
            "dimension": dim,
            "nlist": finalized_nlist,
            "nprobe": int(args.nprobe) if index_type == "ivf_flat" else None,
            "doc_start": shard_start,
            "doc_end": total_docs,
        }
        if doc_file is not None:
            doc_file.close()
        if offsets_file is not None:
            offsets_file.close()
        index = None
        doc_file = None
        offsets_file = None
        dense_cfg_path = None
        shard_dir = None
        shard_doc_count = 0
        shard_nlist = None
        return shard_info

    for batch in _iter_doc_batches(
        args.corpus_path,
        max_docs=max_docs,
        batch_size=int(args.batch_size),
        skip_docs=total_docs,
    ):
        emb = model.encode(
            [_combine_text(doc) for doc in batch],
            batch_size=int(args.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        batch_offset = 0
        while batch_offset < len(batch):
            if shard_dir is None:
                shard_index += 1
                shard_dir = args.output_root / f"shard_{shard_index:03d}"
                _ensure_dir(shard_dir)
                doc_file = (shard_dir / "docstore.jsonl").open("wb")
                offsets_file = (shard_dir / "docstore_offsets.bin").open("wb")
                dense_cfg_path = shard_dir / "dense_config.json"
                shard_start = total_docs

            remaining_capacity = shard_size - shard_doc_count
            take = min(remaining_capacity, len(batch) - batch_offset)
            current_docs = batch[batch_offset : batch_offset + take]
            current_emb = emb[batch_offset : batch_offset + take]

            for doc in current_docs:
                _write_doc_row(doc_file, offsets_file, doc)
            _append_embeddings(current_emb)
            total_docs += take
            shard_doc_count += take
            batch_offset += take

            if shard_doc_count >= shard_size:
                assert shard_dir is not None
                shard_info = finalize_current_shard()
                if shard_info is not None:
                    shards.append(shard_info)

    if shard_dir is not None:
        shard_info = finalize_current_shard()
        if shard_info is not None:
            shards.append(shard_info)

    manifest = {
        "index_format": "dense_sharded",
        "index_type": index_type,
        "embedding_model": args.embedding_model,
        "device": args.device,
        "dimension": dim,
        "total_docs": total_docs,
        "shard_size": shard_size,
        "nlist": int(args.ivf_nlist) if index_type == "ivf_flat" else None,
        "nprobe": int(args.nprobe) if index_type == "ivf_flat" else None,
        "num_workers": min(len(shards), cpu_count() or 1) if shards else 1,
        "shards": shards,
        "corpus_path": str(args.corpus_path.resolve()),
    }
    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "index_type": index_type,
                "total_docs": total_docs,
                "num_shards": len(shards),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
