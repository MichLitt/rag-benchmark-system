"""Tests for Document schema migration (A0).

Verifies backward compatibility of docstore serialization and LazyDocstore thread-safety.
"""
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.retrieval.docstore import (
    build_docstore_offsets,
    load_docstore,
    save_docstore,
    LazyDocstore,
)
from src.types import Document, ScoredDocument


class TestDocstoreRoundTrip(unittest.TestCase):
    def test_round_trip_all_new_fields(self) -> None:
        doc = Document(
            doc_id="d1",
            text="some text here",
            title="My Title",
            page_start=3,
            page_end=4,
            section="Introduction",
            source="test.pdf",
            extra_metadata={"chunk_index": 0, "foo": "bar"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.jsonl"
            save_docstore(path, [doc])
            loaded = load_docstore(path)
        self.assertEqual(len(loaded), 1)
        result = loaded[0]
        self.assertEqual(result.doc_id, "d1")
        self.assertEqual(result.text, "some text here")
        self.assertEqual(result.title, "My Title")
        self.assertEqual(result.page_start, 3)
        self.assertEqual(result.page_end, 4)
        self.assertEqual(result.section, "Introduction")
        self.assertEqual(result.source, "test.pdf")
        self.assertEqual(result.extra_metadata["chunk_index"], 0)
        self.assertEqual(result.extra_metadata["foo"], "bar")

    def test_round_trip_none_fields_omitted(self) -> None:
        """None fields should not appear in the JSONL output."""
        doc = Document(doc_id="d2", text="text", title="Title")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "docs.jsonl"
            save_docstore(path, [doc])
            raw = path.read_text(encoding="utf-8")
        row = json.loads(raw.strip())
        self.assertNotIn("page_start", row)
        self.assertNotIn("page_end", row)
        self.assertNotIn("section", row)
        self.assertNotIn("source", row)
        self.assertNotIn("extra_metadata", row)


class TestBackwardCompatibility(unittest.TestCase):
    def test_old_format_jsonl_loads_cleanly(self) -> None:
        """JSONL with only doc_id/title/text (pre-migration format) should load fine."""
        old_jsonl = (
            '{"doc_id": "x1", "title": "Old Doc", "text": "legacy text"}\n'
            '{"doc_id": "x2", "title": "", "text": "another"}\n'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "old.jsonl"
            path.write_text(old_jsonl, encoding="utf-8")
            docs = load_docstore(path)

        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].doc_id, "x1")
        self.assertEqual(docs[0].title, "Old Doc")
        self.assertEqual(docs[0].text, "legacy text")
        self.assertIsNone(docs[0].page_start)
        self.assertIsNone(docs[0].page_end)
        self.assertIsNone(docs[0].section)
        self.assertIsNone(docs[0].source)
        self.assertEqual(docs[0].extra_metadata, {})


class TestLazyDocstoreThreadSafety(unittest.TestCase):
    def _build_store(self, tmpdir: str, n: int) -> tuple[Path, Path]:
        docs = [
            Document(doc_id=f"doc_{i}", text=f"text_{i}", title=f"title_{i}")
            for i in range(n)
        ]
        docstore_path = Path(tmpdir) / "docs.jsonl"
        offsets_path = Path(tmpdir) / "docs.offsets"
        save_docstore(docstore_path, docs)
        build_docstore_offsets(docstore_path, offsets_path)
        return docstore_path, offsets_path

    def test_concurrent_get_returns_correct_docs(self) -> None:
        """8 threads reading random indexes concurrently should all succeed."""
        n = 100
        with tempfile.TemporaryDirectory() as tmpdir:
            docstore_path, offsets_path = self._build_store(tmpdir, n)
            store = LazyDocstore(docstore_path, offsets_path)

            indexes = list(range(n)) * 3  # 300 reads total
            results: list[Document] = []
            errors: list[Exception] = []

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(store.get, i): i for i in indexes}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        errors.append(exc)

        self.assertEqual(len(errors), 0, f"Errors during concurrent get: {errors}")
        self.assertEqual(len(results), 300)
        # Spot-check correctness
        for doc in results:
            idx = int(doc.doc_id.split("_")[1])
            self.assertEqual(doc.text, f"text_{idx}")

    def test_close_is_noop(self) -> None:
        n = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            docstore_path, offsets_path = self._build_store(tmpdir, n)
            store = LazyDocstore(docstore_path, offsets_path)
            store.close()
            # Can still call get() after close()
            doc = store.get(0)
            self.assertEqual(doc.doc_id, "doc_0")


class TestScoredDocument(unittest.TestCase):
    def test_construction_and_access(self) -> None:
        doc = Document(doc_id="s1", text="hello", title="T")
        scored = ScoredDocument(document=doc, score=0.92, rank=1)
        self.assertEqual(scored.document.doc_id, "s1")
        self.assertAlmostEqual(scored.score, 0.92)
        self.assertEqual(scored.rank, 1)

    def test_frozen(self) -> None:
        doc = Document(doc_id="s2", text="x", title="Y")
        scored = ScoredDocument(document=doc, score=0.5, rank=2)
        with self.assertRaises(Exception):
            scored.score = 0.1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
