import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.types import Document
from src.retrieval.title_prefilter import DenseShardedTitlePrefilterRetriever


def _load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDenseRetriever:
    def __init__(self, *args, **kwargs) -> None:
        self.queries: list[tuple[list[str], int]] = []
        self.nprobe = None
        self.num_workers = 1

    def retrieve_many(self, queries: list[str], top_k: int) -> list[list[Document]]:
        self.queries.append((list(queries), top_k))
        return [
            [
                Document(doc_id="d1", title="Alpha", text="alpha"),
                Document(doc_id="d2", title="Gamma", text="gamma"),
                Document(doc_id="d3", title="Beta", text="beta"),
                Document(doc_id="d4", title="Delta", text="delta"),
            ]
            for _ in queries
        ]

    def close(self) -> None:
        return None


class _FakeTitleRetriever:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return [Document(doc_id="beta", title="Beta", text="Beta")]


class TitlePrefilterTests(unittest.TestCase):
    @patch("src.retrieval.title_prefilter.BM25Retriever", _FakeTitleRetriever)
    @patch("src.retrieval.title_prefilter.ShardedFaissDenseRetriever", _FakeDenseRetriever)
    def test_title_prefilter_promotes_matched_title_from_deeper_dense_candidates(self):
        retriever = DenseShardedTitlePrefilterRetriever(
            manifest_path="dummy-manifest.json",
            title_prefilter_bm25_path="dummy-bm25.pkl",
            title_prefilter_docstore_path="dummy-docstore.jsonl",
            title_prefilter_k=5,
            dense_probe_top_k=4,
        )
        try:
            docs = retriever.retrieve("question", top_k=2)
            self.assertEqual([doc.title for doc in docs], ["Beta", "Alpha"])
            self.assertEqual(retriever._dense.queries[0][1], 4)
        finally:
            retriever.close()

    def test_title_bm25_builder_writes_manifest_and_unique_titles(self):
        builder = _load_script_module("build_title_bm25_index.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "corpus.jsonl"
            corpus_path.write_text(
                "\n".join(
                    [
                        json.dumps({"doc_id": "d1", "title": "Alpha", "text": "alpha 1"}),
                        json.dumps({"doc_id": "d2", "title": "Alpha", "text": "alpha 2"}),
                        json.dumps({"doc_id": "d3", "title": "Beta", "text": "beta 1"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "title_index"
            argv = [
                "build_title_bm25_index.py",
                "--corpus-path",
                str(corpus_path),
                "--output-dir",
                str(output_dir),
                "--representative-docids-per-title",
                "2",
            ]
            with patch("sys.argv", argv):
                builder.main()

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_unique_titles"], 2)
            metadata_rows = [
                json.loads(line)
                for line in (output_dir / "title_metadata.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            alpha_row = next(row for row in metadata_rows if row["normalized_title"] == "alpha")
            self.assertEqual(alpha_row["chunk_count"], 2)
            self.assertEqual(alpha_row["representative_doc_ids"], ["d1", "d2"])


if __name__ == "__main__":
    unittest.main()
