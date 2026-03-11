import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_builder_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "build_dense_sharded_index.py"
    spec = importlib.util.spec_from_file_location("build_dense_sharded_index", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder_module()


def _write_completed_shard(root: Path, shard_id: int, *, num_docs: int, index_type: str = "ivf_flat") -> Path:
    shard_dir = root / f"shard_{shard_id:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "faiss.index").write_bytes(b"index")
    (shard_dir / "docstore.jsonl").write_bytes(b'{"doc_id":"d1","title":"t","text":"x"}\n' * num_docs)
    (shard_dir / "docstore_offsets.bin").write_bytes(b"\x00" * (num_docs * 8))
    (shard_dir / "dense_config.json").write_text(
        json.dumps(
            {
                "embedding_model": "fake-model",
                "index_type": index_type,
                "dimension": 384,
                "num_docs": num_docs,
                "nlist": 128 if index_type == "ivf_flat" else None,
                "nprobe": 16 if index_type == "ivf_flat" else None,
            }
        ),
        encoding="utf-8",
    )
    return shard_dir


class BuildDenseShardedIndexTests(unittest.TestCase):
    def test_prepare_resume_state_reuses_completed_shards_and_cleans_partial_trailing_shard(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_completed_shard(root, 0, num_docs=2)
            _write_completed_shard(root, 1, num_docs=2)

            partial = root / "shard_002"
            partial.mkdir()
            (partial / "docstore.jsonl").write_bytes(b"")
            (partial / "docstore_offsets.bin").write_bytes(b"")

            shards, total_docs, next_shard_id = builder._prepare_resume_state(
                root,
                dimension=384,
                index_type="ivf_flat",
                nprobe=16,
            )

            self.assertEqual(next_shard_id, 2)
            self.assertEqual(total_docs, 4)
            self.assertEqual([int(shard["shard_id"]) for shard in shards], [0, 1])
            self.assertEqual([int(shard["doc_start"]) for shard in shards], [0, 2])
            self.assertEqual([int(shard["doc_end"]) for shard in shards], [2, 4])
            self.assertFalse(partial.exists())

    def test_prepare_resume_state_requires_contiguous_completed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_completed_shard(root, 1, num_docs=2)

            with self.assertRaisesRegex(ValueError, "expected shard_000"):
                builder._prepare_resume_state(
                    root,
                    dimension=384,
                    index_type="ivf_flat",
                    nprobe=16,
                )


if __name__ == "__main__":
    unittest.main()
