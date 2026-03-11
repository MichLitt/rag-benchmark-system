import unittest

import numpy as np

from src.reranking.cross_encoder import rerank_documents_from_scores
from src.types import Document


class RerankerBlendingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.docs = [
            Document(doc_id="d1", title="First", text="a"),
            Document(doc_id="d2", title="Second", text="b"),
            Document(doc_id="d3", title="Third", text="c"),
        ]

    def test_pure_reranker_order_kept_when_weight_zero(self):
        reranked = rerank_documents_from_scores(
            docs=self.docs,
            scores=np.asarray([0.1, 0.9, 0.2], dtype=np.float32),
            top_k=3,
            retriever_rank_weight=0.0,
        )

        self.assertEqual([doc.doc_id for doc in reranked], ["d2", "d3", "d1"])

    def test_retriever_order_kept_when_weight_one(self):
        reranked = rerank_documents_from_scores(
            docs=self.docs,
            scores=np.asarray([0.1, 0.9, 0.2], dtype=np.float32),
            top_k=3,
            retriever_rank_weight=1.0,
        )

        self.assertEqual([doc.doc_id for doc in reranked], ["d1", "d2", "d3"])

    def test_blending_can_preserve_high_retriever_rank_in_top_k(self):
        reranked = rerank_documents_from_scores(
            docs=self.docs,
            scores=np.asarray([0.4, 0.9, 0.8], dtype=np.float32),
            top_k=2,
            retriever_rank_weight=0.7,
            rank_fusion_k=1,
        )

        self.assertEqual([doc.doc_id for doc in reranked], ["d1", "d2"])


if __name__ == "__main__":
    unittest.main()
