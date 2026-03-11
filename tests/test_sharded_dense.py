import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from src.retrieval.docstore import LazyDocstore, build_docstore_offsets
from src.retrieval.factory import build_retriever
from src.retrieval.sharded_dense import ShardedFaissDenseRetriever


class _FakeSentenceTransformer:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(self, texts, **kwargs):
        vectors = {
            "query:alpha": np.array([1.0, 0.0], dtype=np.float32),
            "query:beta": np.array([0.0, 1.0], dtype=np.float32),
        }
        return np.vstack([vectors[text] for text in texts]).astype(np.float32, copy=False)


def _write_docstore(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _write_flat_index(path: Path, vectors: list[list[float]]) -> Path:
    index = faiss.IndexFlatIP(len(vectors[0]))
    index.add(np.asarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(path))
    return path


def _write_ivf_index(path: Path, vectors: list[list[float]], nlist: int = 1) -> Path:
    emb = np.asarray(vectors, dtype=np.float32)
    quantizer = faiss.IndexFlatIP(emb.shape[1])
    index = faiss.IndexIVFFlat(quantizer, emb.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    index.nprobe = 1
    faiss.write_index(index, str(path))
    return path


def _write_dense_cfg(path: Path, *, index_type: str = "flat", nlist: int | None = None, nprobe: int | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "embedding_model": "fake-model",
                "index_type": index_type,
                "dimension": 2,
                "num_docs": 2,
                "nlist": nlist,
                "nprobe": nprobe,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _write_manifest(root: Path, shards: list[dict], *, index_type: str) -> Path:
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "index_format": "dense_sharded",
                "index_type": index_type,
                "embedding_model": "fake-model",
                "dimension": 2,
                "nprobe": 4 if index_type == "ivf_flat" else None,
                "shard_size": 2,
                "total_docs": sum(int(shard["doc_end"]) - int(shard["doc_start"]) for shard in shards),
                "shards": shards,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


class ShardedDenseTests(unittest.TestCase):
    def test_lazy_docstore_reads_rows_by_offset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            docstore_path = _write_docstore(
                root / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "text a"},
                    {"doc_id": "d2", "title": "Beta", "text": "text b"},
                ],
            )
            offsets_path = root / "docstore_offsets.bin"
            count = build_docstore_offsets(docstore_path, offsets_path)

            store = LazyDocstore(docstore_path, offsets_path)
            try:
                self.assertEqual(count, 2)
                self.assertEqual(store.get(0).doc_id, "d1")
                self.assertEqual(store.get(1).title, "Beta")
            finally:
                store.close()

    @patch("src.retrieval.sharded_dense.SentenceTransformer", _FakeSentenceTransformer)
    def test_sharded_dense_merges_top_k_across_shards(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard0 = root / "shard_000"
            shard1 = root / "shard_001"
            shard0.mkdir()
            shard1.mkdir()

            docstore0 = _write_docstore(
                shard0 / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "s0 d1"},
                    {"doc_id": "d2", "title": "Beta", "text": "s0 d2"},
                ],
            )
            offsets0 = shard0 / "docstore_offsets.bin"
            build_docstore_offsets(docstore0, offsets0)
            _write_flat_index(shard0 / "faiss.index", [[0.8, 0.2], [0.1, 0.9]])
            _write_dense_cfg(shard0 / "dense_config.json")

            docstore1 = _write_docstore(
                shard1 / "docstore.jsonl",
                [
                    {"doc_id": "d3", "title": "Gamma", "text": "s1 d1"},
                    {"doc_id": "d4", "title": "Delta", "text": "s1 d2"},
                ],
            )
            offsets1 = shard1 / "docstore_offsets.bin"
            build_docstore_offsets(docstore1, offsets1)
            _write_flat_index(shard1 / "faiss.index", [[0.95, 0.05], [0.2, 0.8]])
            _write_dense_cfg(shard1 / "dense_config.json")

            manifest_path = _write_manifest(
                root,
                [
                    {
                        "index_type": "flat",
                        "faiss_index_path": str((shard0 / "faiss.index").resolve()),
                        "docstore_path": str(docstore0.resolve()),
                        "docstore_offsets_path": str(offsets0.resolve()),
                        "dense_config_path": str((shard0 / "dense_config.json").resolve()),
                        "dimension": 2,
                        "doc_start": 0,
                        "doc_end": 2,
                    },
                    {
                        "index_type": "flat",
                        "faiss_index_path": str((shard1 / "faiss.index").resolve()),
                        "docstore_path": str(docstore1.resolve()),
                        "docstore_offsets_path": str(offsets1.resolve()),
                        "dense_config_path": str((shard1 / "dense_config.json").resolve()),
                        "dimension": 2,
                        "doc_start": 2,
                        "doc_end": 4,
                    },
                ],
                index_type="flat",
            )

            retriever = ShardedFaissDenseRetriever(manifest_path)
            try:
                docs = retriever.retrieve("query:alpha", top_k=3)
                self.assertEqual([doc.doc_id for doc in docs], ["d3", "d1", "d4"])
            finally:
                retriever.close()

    @patch("src.retrieval.sharded_dense.SentenceTransformer", _FakeSentenceTransformer)
    def test_retrieve_many_returns_results_per_query(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard = root / "shard_000"
            shard.mkdir()
            docstore = _write_docstore(
                shard / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "text a"},
                    {"doc_id": "d2", "title": "Beta", "text": "text b"},
                ],
            )
            offsets = shard / "docstore_offsets.bin"
            build_docstore_offsets(docstore, offsets)
            _write_flat_index(shard / "faiss.index", [[0.9, 0.1], [0.1, 0.9]])
            _write_dense_cfg(shard / "dense_config.json")
            manifest_path = _write_manifest(
                root,
                [
                    {
                        "index_type": "flat",
                        "faiss_index_path": str((shard / "faiss.index").resolve()),
                        "docstore_path": str(docstore.resolve()),
                        "docstore_offsets_path": str(offsets.resolve()),
                        "dense_config_path": str((shard / "dense_config.json").resolve()),
                        "dimension": 2,
                        "doc_start": 0,
                        "doc_end": 2,
                    }
                ],
                index_type="flat",
            )

            retriever = ShardedFaissDenseRetriever(manifest_path)
            try:
                docs = retriever.retrieve_many(["query:alpha", "query:beta"], top_k=1)
                self.assertEqual(len(docs), 2)
                self.assertEqual([doc.doc_id for doc in docs[0]], ["d1"])
                self.assertEqual([doc.doc_id for doc in docs[1]], ["d2"])
            finally:
                retriever.close()

    @patch("src.retrieval.sharded_dense.SentenceTransformer", _FakeSentenceTransformer)
    def test_parallel_and_serial_flat_search_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard0 = root / "shard_000"
            shard1 = root / "shard_001"
            shard0.mkdir()
            shard1.mkdir()
            docstore0 = _write_docstore(
                shard0 / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "s0 d1"},
                    {"doc_id": "d2", "title": "Beta", "text": "s0 d2"},
                ],
            )
            offsets0 = shard0 / "docstore_offsets.bin"
            build_docstore_offsets(docstore0, offsets0)
            _write_flat_index(shard0 / "faiss.index", [[0.8, 0.2], [0.1, 0.9]])
            _write_dense_cfg(shard0 / "dense_config.json")

            docstore1 = _write_docstore(
                shard1 / "docstore.jsonl",
                [
                    {"doc_id": "d3", "title": "Gamma", "text": "s1 d1"},
                    {"doc_id": "d4", "title": "Delta", "text": "s1 d2"},
                ],
            )
            offsets1 = shard1 / "docstore_offsets.bin"
            build_docstore_offsets(docstore1, offsets1)
            _write_flat_index(shard1 / "faiss.index", [[0.95, 0.05], [0.2, 0.8]])
            _write_dense_cfg(shard1 / "dense_config.json")

            manifest_path = _write_manifest(
                root,
                [
                    {
                        "index_type": "flat",
                        "faiss_index_path": str((shard0 / "faiss.index").resolve()),
                        "docstore_path": str(docstore0.resolve()),
                        "docstore_offsets_path": str(offsets0.resolve()),
                        "dense_config_path": str((shard0 / "dense_config.json").resolve()),
                        "dimension": 2,
                        "doc_start": 0,
                        "doc_end": 2,
                    },
                    {
                        "index_type": "flat",
                        "faiss_index_path": str((shard1 / "faiss.index").resolve()),
                        "docstore_path": str(docstore1.resolve()),
                        "docstore_offsets_path": str(offsets1.resolve()),
                        "dense_config_path": str((shard1 / "dense_config.json").resolve()),
                        "dimension": 2,
                        "doc_start": 2,
                        "doc_end": 4,
                    },
                ],
                index_type="flat",
            )

            serial = ShardedFaissDenseRetriever(manifest_path, num_workers=1)
            parallel = ShardedFaissDenseRetriever(manifest_path, num_workers=2)
            try:
                serial_docs = serial.retrieve_many(["query:alpha", "query:beta"], top_k=2)
                parallel_docs = parallel.retrieve_many(["query:alpha", "query:beta"], top_k=2)
                self.assertEqual(
                    [[doc.doc_id for doc in docs] for docs in serial_docs],
                    [[doc.doc_id for doc in docs] for docs in parallel_docs],
                )
            finally:
                serial.close()
                parallel.close()

    @patch("src.retrieval.sharded_dense.SentenceTransformer", _FakeSentenceTransformer)
    def test_ivf_manifest_loads_and_applies_nprobe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard = root / "shard_000"
            shard.mkdir()
            docstore = _write_docstore(
                shard / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "text a"},
                    {"doc_id": "d2", "title": "Beta", "text": "text b"},
                ],
            )
            offsets = shard / "docstore_offsets.bin"
            build_docstore_offsets(docstore, offsets)
            _write_ivf_index(shard / "faiss.index", [[0.9, 0.1], [0.1, 0.9]])
            _write_dense_cfg(shard / "dense_config.json", index_type="ivf_flat", nlist=1, nprobe=4)
            manifest_path = _write_manifest(
                root,
                [
                    {
                        "index_type": "ivf_flat",
                        "faiss_index_path": str((shard / "faiss.index").resolve()),
                        "docstore_path": str(docstore.resolve()),
                        "docstore_offsets_path": str(offsets.resolve()),
                        "dense_config_path": str((shard / "dense_config.json").resolve()),
                        "dimension": 2,
                        "nlist": 1,
                        "nprobe": 4,
                        "doc_start": 0,
                        "doc_end": 2,
                    }
                ],
                index_type="ivf_flat",
            )

            retriever = ShardedFaissDenseRetriever(manifest_path)
            try:
                self.assertEqual(retriever.retrieval_mode, "ivf_flat")
                self.assertEqual(retriever.nprobe, 4)
                docs = retriever.retrieve("query:alpha", top_k=1)
                self.assertEqual([doc.doc_id for doc in docs], ["d1"])
            finally:
                retriever.close()

    @patch("src.retrieval.sharded_dense.SentenceTransformer", _FakeSentenceTransformer)
    def test_factory_builds_dense_sharded_retriever(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard = root / "shard_000"
            shard.mkdir()
            docstore = _write_docstore(
                shard / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "text a"},
                    {"doc_id": "d2", "title": "Beta", "text": "text b"},
                ],
            )
            offsets = shard / "docstore_offsets.bin"
            build_docstore_offsets(docstore, offsets)
            _write_flat_index(shard / "faiss.index", [[0.8, 0.2], [0.1, 0.9]])
            _write_dense_cfg(shard / "dense_config.json")
            manifest_path = _write_manifest(
                root,
                [
                    {
                        "index_type": "flat",
                        "faiss_index_path": str((shard / "faiss.index").resolve()),
                        "docstore_path": str(docstore.resolve()),
                        "docstore_offsets_path": str(offsets.resolve()),
                        "dense_config_path": str((shard / "dense_config.json").resolve()),
                        "dimension": 2,
                        "doc_start": 0,
                        "doc_end": 2,
                    }
                ],
                index_type="flat",
            )

            retriever = build_retriever(
                {
                    "retrieval": {
                        "mode": "dense_sharded",
                        "dense_shards_manifest_path": str(manifest_path),
                        "nprobe": 8,
                        "num_workers": 1,
                    }
                },
                corpus=[],
            )
            try:
                self.assertIsInstance(retriever, ShardedFaissDenseRetriever)
                self.assertEqual(retriever.num_workers, 1)
            finally:
                retriever.close()

    @patch("src.retrieval.sharded_dense.SentenceTransformer", _FakeSentenceTransformer)
    def test_legacy_manifest_without_ann_fields_falls_back_to_flat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            shard = root / "shard_000"
            shard.mkdir()
            docstore = _write_docstore(
                shard / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "text a"},
                    {"doc_id": "d2", "title": "Beta", "text": "text b"},
                ],
            )
            offsets = shard / "docstore_offsets.bin"
            build_docstore_offsets(docstore, offsets)
            _write_flat_index(shard / "faiss.index", [[0.8, 0.2], [0.1, 0.9]])
            _write_dense_cfg(shard / "dense_config.json")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "index_type": "dense_sharded_flat",
                        "embedding_model": "fake-model",
                        "total_docs": 2,
                        "shard_size": 2,
                        "shards": [
                            {
                                "faiss_index_path": str((shard / "faiss.index").resolve()),
                                "docstore_path": str(docstore.resolve()),
                                "docstore_offsets_path": str(offsets.resolve()),
                                "dense_config_path": str((shard / "dense_config.json").resolve()),
                                "doc_start": 0,
                                "doc_end": 2,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            retriever = ShardedFaissDenseRetriever(manifest_path)
            try:
                self.assertEqual(retriever.retrieval_mode, "flat")
                docs = retriever.retrieve("query:alpha", top_k=1)
                self.assertEqual([doc.doc_id for doc in docs], ["d1"])
            finally:
                retriever.close()


if __name__ == "__main__":
    unittest.main()
