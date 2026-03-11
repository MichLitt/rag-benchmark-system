import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_hotpot_retrieval import _load_or_build_coverage_titles


def _write_docstore(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class HotpotRetrievalEvalTests(unittest.TestCase):
    def test_coverage_cache_builds_and_reuses_titles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            docstore = _write_docstore(
                root / "docstore.jsonl",
                [
                    {"doc_id": "d1", "title": "Alpha", "text": "a"},
                    {"doc_id": "d2", "title": "Beta", "text": "b"},
                ],
            )
            cache_path = root / "titles_cache.json.gz"

            titles, source_label = _load_or_build_coverage_titles([docstore], cache_path, refresh=False)
            self.assertEqual(titles, {"alpha", "beta"})
            self.assertIn(str(docstore.resolve()), source_label)
            self.assertTrue(cache_path.exists())

            _write_docstore(
                docstore,
                [
                    {"doc_id": "d1", "title": "Gamma", "text": "c"},
                ],
            )

            cached_titles, _ = _load_or_build_coverage_titles([docstore], cache_path, refresh=False)
            self.assertEqual(cached_titles, {"alpha", "beta"})

            refreshed_titles, _ = _load_or_build_coverage_titles([docstore], cache_path, refresh=True)
            self.assertEqual(refreshed_titles, {"gamma"})


if __name__ == "__main__":
    unittest.main()
