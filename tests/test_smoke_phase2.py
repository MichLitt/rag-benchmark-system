"""Phase 2 smoke tests.

Covers A0/A1/A2/A3 end-to-end without requiring large model downloads,
pre-built FAISS indexes, or real PDF files.

- A0: types + docstore round-trip with new fields
- A1: TokenAwareChunker on synthetic text; PdfParser on a minimal in-memory PDF
- A2: FastAPI server health + retrieve endpoints (toy keyword retriever, no FAISS)
- A3: CitationEvaluator with a stub scorer; nli_score_faithfulness integration
"""
from __future__ import annotations

import io
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation.citation import CitationEvaluator, CitationResult
from src.evaluation.faithfulness import nli_score_faithfulness, FaithfulnessResult
from src.evaluation.hhem_scorer import HHEMResult, HHEMScorer
from src.ingestion.chunker import TokenAwareChunker
from src.ingestion.factory import get_parser
from src.retrieval.docstore import build_docstore_offsets, load_docstore, save_docstore, LazyDocstore
from src.types import Document, RunExampleResult, ScoredDocument


# ---------------------------------------------------------------------------
# A0 smoke
# ---------------------------------------------------------------------------

class SmokeA0Types(unittest.TestCase):
    def test_document_new_fields_default_none(self) -> None:
        doc = Document(doc_id="x", text="hello", title="T")
        self.assertIsNone(doc.page_start)
        self.assertIsNone(doc.page_end)
        self.assertIsNone(doc.section)
        self.assertIsNone(doc.source)
        self.assertEqual(doc.extra_metadata, {})

    def test_document_with_all_fields(self) -> None:
        doc = Document(
            doc_id="x", text="hello", title="T",
            page_start=1, page_end=2,
            section="Intro", source="doc.pdf",
            extra_metadata={"chunk_index": 0},
        )
        self.assertEqual(doc.page_start, 1)
        self.assertEqual(doc.extra_metadata["chunk_index"], 0)

    def test_scored_document(self) -> None:
        doc = Document(doc_id="d1", text="x", title="Y")
        sd = ScoredDocument(document=doc, score=0.85, rank=1)
        self.assertEqual(sd.document.doc_id, "d1")
        self.assertAlmostEqual(sd.score, 0.85)

    def test_run_example_result_nli_fields_default_none(self) -> None:
        r = RunExampleResult(
            query_id="q1", predicted_answer="ans", gold_answers=["ans"],
            retrieved_doc_ids=[], retrieved_titles=[], unique_retrieved_titles=0,
            retrieval_latency_ms=0.0, rerank_latency_ms=0.0,
            generation_latency_ms=0.0, approx_input_tokens=0,
            approx_output_tokens=0, is_em=True, f1=1.0, recall_at_k=1.0,
            raw_candidate_count=0, dedup_candidate_count=0,
            duplicate_candidates_removed=0,
        )
        self.assertIsNone(r.nli_answer_attribution_rate)
        self.assertIsNone(r.nli_supporting_passage_hit)
        self.assertIsNone(r.nli_page_grounding_accuracy)

    def test_docstore_round_trip_new_fields(self) -> None:
        doc = Document(
            doc_id="d1", text="some text", title="Title",
            page_start=3, page_end=4, section="Sec", source="a.pdf",
            extra_metadata={"k": "v"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store.jsonl"
            save_docstore(p, [doc])
            loaded = load_docstore(p)
        self.assertEqual(loaded[0].page_start, 3)
        self.assertEqual(loaded[0].section, "Sec")
        self.assertEqual(loaded[0].extra_metadata["k"], "v")

    def test_docstore_backward_compat(self) -> None:
        old_line = '{"doc_id": "old1", "title": "Old", "text": "legacy"}\n'
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "old.jsonl"
            p.write_text(old_line, encoding="utf-8")
            docs = load_docstore(p)
        self.assertEqual(len(docs), 1)
        self.assertIsNone(docs[0].page_start)
        self.assertEqual(docs[0].extra_metadata, {})

    def test_lazy_docstore_get(self) -> None:
        docs = [Document(doc_id=f"d{i}", text=f"t{i}", title="") for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            dp = Path(tmp) / "d.jsonl"
            op = Path(tmp) / "d.offsets"
            save_docstore(dp, docs)
            build_docstore_offsets(dp, op)
            store = LazyDocstore(dp, op)
            result = store.get(2)
        self.assertEqual(result.doc_id, "d2")
        self.assertEqual(result.text, "t2")


# ---------------------------------------------------------------------------
# A1 smoke
# ---------------------------------------------------------------------------

class SmokeA1Chunker(unittest.TestCase):
    def test_short_text_returns_single_doc(self) -> None:
        doc = Document(doc_id="d1", text="short text", title="T")
        chunker = TokenAwareChunker(chunk_size=256, overlap=32)
        chunks = chunker.chunk(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "d1")

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        # Create text that's clearly > 256 tokens
        long_text = " ".join([f"word{i}" for i in range(600)])
        doc = Document(doc_id="long", text=long_text, title="T",
                       page_start=1, page_end=1, source="test.pdf")
        chunker = TokenAwareChunker(chunk_size=256, overlap=32)
        chunks = chunker.chunk(doc)
        self.assertGreater(len(chunks), 1)
        # All chunks preserve parent metadata
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.page_start, 1)
            self.assertEqual(chunk.source, "test.pdf")
            self.assertEqual(chunk.extra_metadata.get("chunk_index"), i)

    def test_chunk_ids_are_unique(self) -> None:
        long_text = " ".join([f"token{i}" for i in range(800)])
        doc = Document(doc_id="base", text=long_text, title="T")
        chunker = TokenAwareChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(doc)
        ids = [c.doc_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), "chunk IDs must be unique")

    def test_invalid_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            TokenAwareChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            TokenAwareChunker(chunk_size=100, overlap=100)

    def test_factory_returns_pdf_parser(self) -> None:
        from src.ingestion.pdf_parser import PdfParser
        parser = get_parser("pdf")
        self.assertIsInstance(parser, PdfParser)

    def test_factory_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            get_parser("ocr")

    def test_pdf_parser_smoke_with_minimal_pdf(self) -> None:
        """Parse a minimal valid PDF generated in-memory."""
        # Build a minimal single-page PDF with embedded text using bytes.
        # This uses the simplest possible PDF structure.
        minimal_pdf = _make_minimal_pdf("Hello from smoke test. This is page one content.")
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "test.pdf"
            pdf_path.write_bytes(minimal_pdf)
            from src.ingestion.pdf_parser import PdfParser
            parser = PdfParser()
            docs = parser.parse(pdf_path)
        # pdfplumber may or may not extract text from this minimal PDF.
        # We only assert no exception was raised and the return type is correct.
        self.assertIsInstance(docs, list)
        for doc in docs:
            self.assertIsInstance(doc, Document)
            self.assertIsNotNone(doc.page_start)


def _make_minimal_pdf(text: str) -> bytes:
    """Build a minimal valid PDF with one page containing the given text.

    This uses the PDF content-stream approach without any external library.
    Suitable only for smoke tests.
    """
    # Escape special PDF string chars
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects = []

    # Object 1: catalog
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # Object 2: pages
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # Object 4: content stream
    stream_content = f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")
    stream = (
        b"4 0 obj\n<< /Length " + str(len(stream_content)).encode() + b" >>\nstream\n"
        + stream_content + b"\nendstream\nendobj\n"
    )
    objects.append(stream)
    # Object 3: page (references content)
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
        b">>\nendobj\n"
    )

    # Build cross-reference table
    header = b"%PDF-1.4\n"
    body = b""
    offsets: list[int] = []
    pos = len(header)
    for obj in objects:
        offsets.append(pos)
        body += obj
        pos += len(obj)

    xref_pos = len(header) + len(body)
    xref = b"xref\n0 5\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )

    return header + body + xref + trailer


# ---------------------------------------------------------------------------
# A2 smoke
# ---------------------------------------------------------------------------

class SmokeA2API(unittest.TestCase):
    """Test the FastAPI endpoints using TestClient (no real index needed)."""

    def _make_client(self):
        from fastapi.testclient import TestClient
        # Import fresh to avoid shared module-level _registry state between tests
        import importlib
        import src.api.server as server_mod
        importlib.reload(server_mod)
        return TestClient(server_mod.app)

    def test_health_returns_ok_with_no_indexes(self) -> None:
        client = self._make_client()
        resp = client.get("/v1/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIsInstance(data["loaded_indexes"], list)

    def test_retrieve_returns_404_for_unknown_index(self) -> None:
        client = self._make_client()
        resp = client.post("/v1/retrieve", json={
            "query": "test query",
            "top_k": 3,
            "index_id": "nonexistent",
        })
        self.assertEqual(resp.status_code, 404)

    def test_retrieve_with_injected_keyword_retriever(self) -> None:
        """Register a toy keyword retriever and verify /v1/retrieve response shape."""
        from src.retrieval.keyword import KeywordRetriever

        docs = [
            Document(doc_id="d1", text="Python is great", title="Doc1",
                     page_start=1, page_end=1, source="a.pdf"),
            Document(doc_id="d2", text="FastAPI is fast", title="Doc2",
                     page_start=2, page_end=2, source="a.pdf"),
            Document(doc_id="d3", text="Retrieval augmented generation", title="Doc3"),
        ]
        retriever = KeywordRetriever(docs)

        import src.api.server as server_mod
        import importlib
        importlib.reload(server_mod)
        server_mod._registry._registry["test"] = retriever

        from fastapi.testclient import TestClient
        client = TestClient(server_mod.app)

        resp = client.post("/v1/retrieve", json={
            "query": "Python FastAPI",
            "top_k": 2,
            "index_id": "test",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["index_id"], "test")
        self.assertEqual(data["query"], "Python FastAPI")
        self.assertIsInstance(data["passages"], list)
        self.assertLessEqual(len(data["passages"]), 2)

        if data["passages"]:
            p = data["passages"][0]
            # Required fields
            for field in ("doc_id", "title", "text", "score", "rank"):
                self.assertIn(field, p)
            # score should be in (0, 1]
            self.assertGreater(p["score"], 0.0)
            self.assertLessEqual(p["score"], 1.0)
            # rank 1 should have highest score
            self.assertEqual(data["passages"][0]["rank"], 1)

    def test_retrieve_response_passes_page_metadata(self) -> None:
        """Passages with page_start/page_end should appear in the response."""
        from src.retrieval.keyword import KeywordRetriever
        docs = [Document(doc_id="p1", text="page content here", title="Doc",
                         page_start=5, page_end=6, source="test.pdf")]
        retriever = KeywordRetriever(docs)

        import src.api.server as server_mod
        import importlib
        importlib.reload(server_mod)
        server_mod._registry._registry["paged"] = retriever

        from fastapi.testclient import TestClient
        client = TestClient(server_mod.app)
        resp = client.post("/v1/retrieve", json={
            "query": "page content",
            "top_k": 1,
            "index_id": "paged",
        })
        self.assertEqual(resp.status_code, 200)
        passages = resp.json()["passages"]
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["page_start"], 5)
        self.assertEqual(passages[0]["page_end"], 6)
        self.assertEqual(passages[0]["source"], "test.pdf")


# ---------------------------------------------------------------------------
# A3 smoke
# ---------------------------------------------------------------------------

class _StubScorer:
    """Stub HHEMScorer that returns pre-set scores without loading any model."""

    def __init__(self, scores: list[float], threshold: float = 0.5) -> None:
        self._scores = scores
        self._threshold = threshold

    def score(self, source: str, summary: str) -> HHEMResult:
        idx = len(self._scores) - 1  # default last
        return HHEMResult(
            score=self._scores[0],
            is_consistent=self._scores[0] >= self._threshold,
        )

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[HHEMResult]:
        return [
            HHEMResult(
                score=self._scores[i % len(self._scores)],
                is_consistent=self._scores[i % len(self._scores)] >= self._threshold,
            )
            for i in range(len(pairs))
        ]


class SmokeA3Citation(unittest.TestCase):
    def _passages(self) -> list[Document]:
        return [
            Document(doc_id="p1", text="The sky is blue.", title="Doc1", page_start=1, page_end=1),
            Document(doc_id="p2", text="Water is wet.", title="Doc2", page_start=2, page_end=2),
            Document(doc_id="p3", text="Grass is green.", title="Doc3"),  # no page metadata
        ]

    def test_all_consistent(self) -> None:
        scorer = _StubScorer([0.8, 0.9, 0.7])
        evaluator = CitationEvaluator(scorer)
        result = evaluator.evaluate("The sky is blue and water is wet.", self._passages())
        self.assertAlmostEqual(result.answer_attribution_rate, 1.0)
        self.assertTrue(result.supporting_passage_hit)
        self.assertEqual(len(result.passage_scores), 3)

    def test_none_consistent(self) -> None:
        scorer = _StubScorer([0.1, 0.2, 0.3])
        evaluator = CitationEvaluator(scorer)
        result = evaluator.evaluate("Some unrelated claim.", self._passages())
        self.assertAlmostEqual(result.answer_attribution_rate, 0.0)
        self.assertFalse(result.supporting_passage_hit)

    def test_partial_consistent(self) -> None:
        # Scores: p1=0.8 (consistent), p2=0.3 (not), p3=0.8 (consistent)
        scorer = _StubScorer([0.8, 0.3, 0.8])
        evaluator = CitationEvaluator(scorer)
        result = evaluator.evaluate("Some answer.", self._passages())
        self.assertAlmostEqual(result.answer_attribution_rate, 2 / 3)
        self.assertTrue(result.supporting_passage_hit)

    def test_page_grounding_accuracy_with_metadata(self) -> None:
        # All 3 passages consistent; p3 has no page_start → accuracy = 2/3
        scorer = _StubScorer([0.9, 0.9, 0.9])
        evaluator = CitationEvaluator(scorer)
        result = evaluator.evaluate("answer", self._passages())
        self.assertIsNotNone(result.page_grounding_accuracy)
        self.assertAlmostEqual(result.page_grounding_accuracy, 2 / 3)

    def test_empty_answer_returns_zero(self) -> None:
        scorer = _StubScorer([0.9])
        evaluator = CitationEvaluator(scorer)
        result = evaluator.evaluate("", self._passages())
        self.assertAlmostEqual(result.answer_attribution_rate, 0.0)
        self.assertFalse(result.supporting_passage_hit)
        self.assertIsNone(result.page_grounding_accuracy)

    def test_empty_passages_returns_zero(self) -> None:
        scorer = _StubScorer([0.9])
        evaluator = CitationEvaluator(scorer)
        result = evaluator.evaluate("some answer", [])
        self.assertAlmostEqual(result.answer_attribution_rate, 0.0)

    def test_nli_score_faithfulness_integration(self) -> None:
        scorer = _StubScorer([0.8, 0.9])
        passages = [
            Document(doc_id="a", text="context A", title=""),
            Document(doc_id="b", text="context B", title=""),
        ]
        result = nli_score_faithfulness(
            answer="some answer",
            context_docs=passages,
            hhem_scorer=scorer,
        )
        self.assertIsInstance(result, FaithfulnessResult)
        self.assertAlmostEqual(result.score, 1.0)  # both consistent
        self.assertIn("nli_attribution", result.reasoning)
        self.assertEqual(result.raw_response, "")

    def test_hhem_result_dataclass(self) -> None:
        r = HHEMResult(score=0.75, is_consistent=True)
        self.assertAlmostEqual(r.score, 0.75)
        self.assertTrue(r.is_consistent)
        self.assertEqual(r.error, "")


if __name__ == "__main__":
    unittest.main()
