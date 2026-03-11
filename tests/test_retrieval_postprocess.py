import unittest

from src.retrieval.postprocess import (
    build_hotpot_gold_diagnostics,
    deduplicate_documents,
    pack_title_diverse_documents,
    select_title_representatives,
)
from src.types import Document


class RetrievalPostprocessTests(unittest.TestCase):
    def test_deduplicate_by_title_preserves_first_hit_order(self):
        docs = [
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d2", title="Alpha", text="a2"),
            Document(doc_id="d3", title="Beta", text="b1"),
        ]

        deduped = deduplicate_documents(docs, "title")

        self.assertEqual([doc.doc_id for doc in deduped], ["d1", "d3"])

    def test_deduplicate_by_doc_id(self):
        docs = [
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d1", title="Alpha copy", text="a2"),
            Document(doc_id="d2", title="Beta", text="b1"),
        ]

        deduped = deduplicate_documents(docs, "doc_id")

        self.assertEqual([doc.doc_id for doc in deduped], ["d1", "d2"])

    def test_deduplicate_off_returns_original_docs(self):
        docs = [
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d2", title="Alpha", text="a2"),
        ]

        deduped = deduplicate_documents(docs, "off")

        self.assertEqual([doc.doc_id for doc in deduped], ["d1", "d2"])

    def test_select_title_representatives_keeps_first_doc_per_title(self):
        docs = [
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d2", title="Alpha", text="a2"),
            Document(doc_id="d3", title="Beta", text="b1"),
        ]

        representatives = select_title_representatives(docs, max_titles=2)

        self.assertEqual([doc.doc_id for doc in representatives], ["d1", "d3"])

    def test_pack_title_diverse_documents_preserves_unique_titles_first(self):
        raw_docs = [
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d2", title="Alpha", text="a2"),
            Document(doc_id="d3", title="Beta", text="b1"),
            Document(doc_id="d4", title="Gamma", text="g1"),
        ]
        ranked_title_docs = [
            Document(doc_id="d3", title="Beta", text="b1"),
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d4", title="Gamma", text="g1"),
        ]

        packed = pack_title_diverse_documents(
            ranked_title_docs=ranked_title_docs,
            raw_candidate_docs=raw_docs,
            top_k=4,
            max_chunks_per_title=2,
            min_unique_titles=3,
        )

        self.assertEqual([doc.doc_id for doc in packed], ["d3", "d1", "d4", "d2"])

    def test_build_hotpot_gold_diagnostics_reports_loss_after_rerank(self):
        gold_titles = ["Alpha", "Beta"]
        raw_docs = [
            Document(doc_id="d1", title="Alpha", text="a1"),
            Document(doc_id="d2", title="Beta", text="b1"),
            Document(doc_id="d3", title="Gamma", text="g1"),
        ]
        deduped_docs = list(raw_docs)
        final_docs = [
            Document(doc_id="d3", title="Gamma", text="g1"),
            Document(doc_id="d1", title="Alpha", text="a1"),
        ]

        diagnostics = build_hotpot_gold_diagnostics(
            gold_titles=gold_titles,
            raw_candidates=raw_docs,
            deduped_candidates=deduped_docs,
            final_docs=final_docs,
        )

        self.assertEqual(diagnostics.gold_titles_in_raw_candidates, ["Alpha", "Beta"])
        self.assertEqual(diagnostics.gold_titles_in_final_top_k, ["Alpha"])
        self.assertEqual(diagnostics.missing_gold_count, 1)
        self.assertFalse(diagnostics.second_gold_found)
        self.assertEqual(
            diagnostics.retrieval_failure_bucket,
            "both_gold_after_dedup_but_lost_after_rerank",
        )


if __name__ == "__main__":
    unittest.main()
