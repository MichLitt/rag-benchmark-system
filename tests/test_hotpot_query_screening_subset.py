import importlib.util
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch


def _load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HotpotQueryScreeningSubsetTests(unittest.TestCase):
    def test_script_builds_subset_in_source_order(self):
        script = _load_script_module("build_hotpot_query_screening_subset.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            qa_path = root / "qa.jsonl"
            taxonomy_examples_path = root / "taxonomy_examples.jsonl"
            taxonomy_summary_path = root / "taxonomy_summary.json"
            output_path = root / "subset.jsonl"

            qa_rows = [
                {"id": "dev_0", "question": "q0"},
                {"id": "dev_1", "question": "q1"},
                {"id": "dev_2", "question": "q2"},
                {"id": "dev_3", "question": "q3"},
            ]
            taxonomy_rows = [
                {"query_id": "dev_0", "subcategory": "query_formulation_gap"},
                {"query_id": "dev_1", "subcategory": "normalization_or_alias_suspect"},
                {"query_id": "dev_2", "subcategory": "budget_limited"},
                {"query_id": "dev_3", "subcategory": "query_formulation_gap"},
            ]
            taxonomy_summary = {
                "top_blockers": [
                    {"subcategory": "query_formulation_gap"},
                    {"subcategory": "normalization_or_alias_suspect"},
                    {"subcategory": "budget_limited"},
                ]
            }

            qa_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qa_rows),
                encoding="utf-8",
            )
            taxonomy_examples_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in taxonomy_rows),
                encoding="utf-8",
            )
            taxonomy_summary_path.write_text(
                json.dumps(taxonomy_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            original_targets = script.DEFAULT_TARGET_BUCKETS
            script.DEFAULT_TARGET_BUCKETS = OrderedDict(
                [
                    ("query_formulation_gap", 2),
                    ("normalization_or_alias_suspect", 1),
                    ("budget_limited", 1),
                ]
            )
            argv = [
                "build_hotpot_query_screening_subset.py",
                "--qa-path",
                str(qa_path),
                "--taxonomy-examples-path",
                str(taxonomy_examples_path),
                "--taxonomy-summary-path",
                str(taxonomy_summary_path),
                "--output-path",
                str(output_path),
            ]
            try:
                with patch("sys.argv", argv):
                    script.main()
            finally:
                script.DEFAULT_TARGET_BUCKETS = original_targets

            selected_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([row["id"] for row in selected_rows], ["dev_0", "dev_1", "dev_2", "dev_3"])

    def test_script_refills_short_bucket_using_blocker_order(self):
        script = _load_script_module("build_hotpot_query_screening_subset.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            qa_path = root / "qa.jsonl"
            taxonomy_examples_path = root / "taxonomy_examples.jsonl"
            taxonomy_summary_path = root / "taxonomy_summary.json"
            output_path = root / "subset.jsonl"

            qa_rows = [
                {"id": "dev_0", "question": "q0"},
                {"id": "dev_1", "question": "q1"},
                {"id": "dev_2", "question": "q2"},
                {"id": "dev_3", "question": "q3"},
            ]
            taxonomy_rows = [
                {"query_id": "dev_0", "subcategory": "query_formulation_gap"},
                {"query_id": "dev_1", "subcategory": "query_formulation_gap"},
                {"query_id": "dev_2", "subcategory": "budget_limited"},
                {"query_id": "dev_3", "subcategory": "query_formulation_gap"},
            ]
            taxonomy_summary = {
                "top_blockers": [
                    {"subcategory": "query_formulation_gap"},
                    {"subcategory": "budget_limited"},
                ]
            }

            qa_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qa_rows),
                encoding="utf-8",
            )
            taxonomy_examples_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in taxonomy_rows),
                encoding="utf-8",
            )
            taxonomy_summary_path.write_text(
                json.dumps(taxonomy_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            original_targets = script.DEFAULT_TARGET_BUCKETS
            script.DEFAULT_TARGET_BUCKETS = OrderedDict(
                [
                    ("query_formulation_gap", 1),
                    ("normalization_or_alias_suspect", 1),
                    ("budget_limited", 1),
                ]
            )
            argv = [
                "build_hotpot_query_screening_subset.py",
                "--qa-path",
                str(qa_path),
                "--taxonomy-examples-path",
                str(taxonomy_examples_path),
                "--taxonomy-summary-path",
                str(taxonomy_summary_path),
                "--output-path",
                str(output_path),
            ]
            try:
                with patch("sys.argv", argv):
                    script.main()
            finally:
                script.DEFAULT_TARGET_BUCKETS = original_targets

            selected_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(selected_rows), 3)
            self.assertEqual([row["id"] for row in selected_rows], ["dev_0", "dev_1", "dev_2"])


if __name__ == "__main__":
    unittest.main()
