import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.hotpot_failure_taxonomy import (
    classify_taxonomy_record,
    merge_qa_fields,
    resolve_query_id,
    summarize_taxonomy,
)


def _load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HotpotFailureTaxonomyTests(unittest.TestCase):
    def test_classify_budget_limited(self):
        record = {
            "id": "q1",
            "question": "question",
            "gold_titles": ["Alpha", "Beta"],
            "retrieved_titles": ["Alpha", "Gamma"],
            "gold_titles_in_raw_candidates": ["Alpha"],
            "retrieval_failure_bucket": "only_one_gold_in_raw",
        }
        result = classify_taxonomy_record(record, dense_probe_titles=["Beta"], sparse_probe_titles=[])
        self.assertEqual(result.subcategory, "budget_limited")

    def test_classify_embedding_confusion(self):
        record = {
            "id": "q1",
            "question": "question",
            "gold_titles": ["Alpha", "Beta"],
            "retrieved_titles": ["Alpha", "Gamma"],
            "gold_titles_in_raw_candidates": ["Alpha"],
            "retrieval_failure_bucket": "only_one_gold_in_raw",
        }
        result = classify_taxonomy_record(record, dense_probe_titles=[], sparse_probe_titles=["Beta"])
        self.assertEqual(result.subcategory, "embedding_confusion")

    def test_classify_normalization_alias_suspect(self):
        record = {
            "id": "q1",
            "question": "question",
            "gold_titles": ["Ed Wood"],
            "retrieved_titles": ["Ed Wood (film)"],
            "gold_titles_in_raw_candidates": [],
            "retrieval_failure_bucket": "no_gold_in_raw",
        }
        result = classify_taxonomy_record(record, dense_probe_titles=[], sparse_probe_titles=[])
        self.assertEqual(result.subcategory, "normalization_or_alias_suspect")

    def test_classify_query_formulation_gap(self):
        record = {
            "id": "q1",
            "question": "question",
            "gold_titles": ["Alpha", "Beta"],
            "retrieved_titles": ["Gamma", "Delta"],
            "gold_titles_in_raw_candidates": [],
            "retrieval_failure_bucket": "no_gold_in_raw",
        }
        result = classify_taxonomy_record(record, dense_probe_titles=[], sparse_probe_titles=[])
        self.assertEqual(result.subcategory, "query_formulation_gap")

    def test_classify_rerank_loss(self):
        record = {
            "id": "q1",
            "question": "question",
            "gold_titles": ["Alpha", "Beta"],
            "retrieved_titles": ["Alpha"],
            "gold_titles_in_raw_candidates": ["Alpha", "Beta"],
            "retrieval_failure_bucket": "both_gold_after_dedup_but_lost_after_rerank",
        }
        result = classify_taxonomy_record(record)
        self.assertEqual(result.subcategory, "rerank_loss")

    def test_summarize_counts(self):
        records = [
            classify_taxonomy_record(
                {
                    "id": "q1",
                    "question": "question",
                    "gold_titles": ["Alpha", "Beta"],
                    "retrieved_titles": ["Alpha"],
                    "gold_titles_in_raw_candidates": ["Alpha"],
                    "retrieval_failure_bucket": "only_one_gold_in_raw",
                },
                dense_probe_titles=["Beta"],
            ),
            classify_taxonomy_record(
                {
                    "id": "q2",
                    "question": "question",
                    "gold_titles": ["Alpha", "Beta"],
                    "retrieved_titles": ["Alpha", "Beta"],
                    "gold_titles_in_raw_candidates": ["Alpha", "Beta"],
                    "retrieval_failure_bucket": "both_gold_in_final",
                }
            ),
        ]
        summary = summarize_taxonomy(records)
        self.assertEqual(summary["total_examples"], 2)
        self.assertEqual(summary["subcategory_counts"]["budget_limited"]["count"], 1)
        self.assertEqual(summary["subcategory_counts"]["resolved"]["count"], 1)

    def test_resolve_query_id_supports_predictions_shape(self):
        self.assertEqual(resolve_query_id({"query_id": "dev_7"}), "dev_7")
        self.assertEqual(resolve_query_id({"id": "dev_8", "query_id": "dev_x"}), "dev_8")

    def test_merge_qa_fields_supports_query_id_records(self):
        merged = merge_qa_fields(
            [
                {
                    "query_id": "dev_1",
                    "retrieval_failure_bucket": "no_gold_in_raw",
                }
            ],
            [
                {
                    "id": "dev_1",
                    "question": "question 1",
                    "gold_titles": ["Alpha", "Beta"],
                }
            ],
        )
        self.assertEqual(merged[0]["id"], "dev_1")
        self.assertEqual(merged[0]["question"], "question 1")
        self.assertEqual(merged[0]["gold_titles"], ["Alpha", "Beta"])

    def test_script_writes_summary_examples_and_report(self):
        script = _load_script_module("analyze_hotpot_failure_taxonomy.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "details.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "q1",
                            "question": "question 1",
                            "gold_titles": ["Alpha", "Beta"],
                            "retrieved_titles": ["Alpha", "Beta"],
                            "gold_titles_in_raw_candidates": ["Alpha", "Beta"],
                            "retrieval_failure_bucket": "both_gold_in_final",
                        },
                        {
                            "id": "q2",
                            "question": "question 2",
                            "gold_titles": ["Alpha", "Beta"],
                            "retrieved_titles": ["Alpha"],
                            "gold_titles_in_raw_candidates": ["Alpha", "Beta"],
                            "retrieval_failure_bucket": "both_gold_after_dedup_but_lost_after_rerank",
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps({"RecallAllGold@k_title": 0.25}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_dir = root / "taxonomy"
            report_path = root / "report.md"
            argv = [
                "analyze_hotpot_failure_taxonomy.py",
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(output_dir),
                "--report-path",
                str(report_path),
            ]
            with patch("sys.argv", argv):
                script.main()

            summary = json.loads((output_dir / "taxonomy_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total_examples"], 2)
            self.assertTrue((output_dir / "taxonomy_examples.jsonl").exists())
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
