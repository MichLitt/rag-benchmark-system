import unittest

from src.analysis.failure_mode import FailureMode, classify_record


class FailureModeTests(unittest.TestCase):
    def test_classify_record_prefers_retrieval_failure_bucket(self):
        result = classify_record(
            {
                "query_id": "q1",
                "f1": 0.0,
                "is_em": False,
                "recall_at_k": 1.0,
                "gold_titles": ["Alpha", "Beta"],
                "retrieved_titles": ["Alpha"],
                "predicted_answer": "",
                "gold_answers": ["answer"],
                "retrieval_failure_bucket": "both_gold_after_dedup_but_lost_after_rerank",
            }
        )

        self.assertEqual(result.failure_mode, FailureMode.LOST_AFTER_RERANK)

    def test_classify_record_marks_generation_failure_when_both_gold_in_final(self):
        result = classify_record(
            {
                "query_id": "q1",
                "f1": 0.0,
                "is_em": False,
                "recall_at_k": 1.0,
                "gold_titles": ["Alpha", "Beta"],
                "retrieved_titles": ["Alpha", "Beta"],
                "predicted_answer": "",
                "gold_answers": ["answer"],
                "retrieval_failure_bucket": "both_gold_in_final",
            }
        )

        self.assertEqual(result.failure_mode, FailureMode.BOTH_GOLD_IN_FINAL)


if __name__ == "__main__":
    unittest.main()
