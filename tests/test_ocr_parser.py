"""Tests for OcrPdfParser and factory 'ocr' mode.

Strategy:
- Factory and interface tests run without Tesseract (mock pytesseract).
- Real-OCR integration test is skipped when Tesseract binary is absent.
"""
from __future__ import annotations

import io
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.chunker import TokenAwareChunker
from src.ingestion.factory import get_parser
from src.ingestion.ocr_parser import OcrPdfParser
from src.ingestion.pdf_parser import PageSpan

fpdf2 = pytest.importorskip("fpdf", reason="fpdf2 not installed")
FPDF = fpdf2.FPDF

_TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_native_pdf(texts: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for text in texts:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in text.split("\n")[:10]:
            pdf.cell(0, 8, text=line[:200], new_x="LMARGIN", new_y="NEXT")
    return pdf.output()


def _native_pdf_to_image_pdf(native_bytes: bytes) -> bytes:
    """Render every page of a native PDF to a PNG image and embed in a new PDF.

    This produces an image-only PDF that requires OCR to extract text, allowing
    OcrPdfParser to be tested end-to-end with real Tesseract.
    """
    import fitz

    src = fitz.open(stream=native_bytes, filetype="pdf")
    dst = fitz.open()
    matrix = fitz.Matrix(2, 2)
    for page in src:
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        img_page = dst.new_page(width=page.rect.width * 2, height=page.rect.height * 2)
        img_page.insert_image(img_page.rect, stream=img_bytes)
    src.close()
    return dst.tobytes()


# ---------------------------------------------------------------------------
# Factory tests (no Tesseract required)
# ---------------------------------------------------------------------------

def test_factory_returns_ocr_parser():
    parser = get_parser("ocr")
    assert isinstance(parser, OcrPdfParser)


def test_factory_pdf_still_works():
    from src.ingestion.pdf_parser import PdfParser
    assert isinstance(get_parser("pdf"), PdfParser)


def test_factory_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown parser mode"):
        get_parser("word")


def test_factory_error_message_includes_both_modes():
    with pytest.raises(ValueError, match="'pdf', 'ocr'"):
        get_parser("unknown")


# ---------------------------------------------------------------------------
# OcrPdfParser unit tests — pytesseract mocked, Tesseract not required
# ---------------------------------------------------------------------------

def test_ocr_parser_returns_page_spans(tmp_path: Path):
    """OcrPdfParser returns PageSpan list with correct page numbers (mocked OCR)."""
    pdf_bytes = _make_native_pdf(["Page one text.", "Page two text."])
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(pdf_bytes)

    mock_ocr = MagicMock(side_effect=["Mocked page one text.", "Mocked page two text."])
    with patch("pytesseract.image_to_string", mock_ocr):
        parser = OcrPdfParser()
        spans = parser.parse(pdf_path)

    assert len(spans) == 2
    assert spans[0].page_num == 1
    assert spans[1].page_num == 2
    assert "Mocked page one" in spans[0].text
    assert "Mocked page two" in spans[1].text


def test_ocr_parser_skips_empty_pages(tmp_path: Path):
    """Pages where OCR returns only whitespace are excluded from results."""
    pdf_bytes = _make_native_pdf(["Page one.", "Page two.", "Page three."])
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # Page 2 returns blank OCR result
    mock_ocr = MagicMock(side_effect=["Real text.", "   \n  ", "More real text."])
    with patch("pytesseract.image_to_string", mock_ocr):
        spans = OcrPdfParser().parse(pdf_path)

    assert len(spans) == 2
    assert spans[0].page_num == 1
    assert spans[1].page_num == 3  # page 2 was skipped; page 3 keeps its original number


def test_ocr_parser_page_numbers_1indexed(tmp_path: Path):
    """First page must have page_num == 1."""
    pdf_bytes = _make_native_pdf(["Only page."])
    pdf_path = tmp_path / "single.pdf"
    pdf_path.write_bytes(pdf_bytes)

    with patch("pytesseract.image_to_string", return_value="Some text here."):
        spans = OcrPdfParser().parse(pdf_path)

    assert len(spans) == 1
    assert spans[0].page_num == 1


def test_ocr_parser_empty_pdf_returns_empty(tmp_path: Path):
    """A PDF where all pages yield empty OCR text returns an empty list."""
    pdf_bytes = _make_native_pdf(["anything"])
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(pdf_bytes)

    with patch("pytesseract.image_to_string", return_value="   "):
        spans = OcrPdfParser().parse(pdf_path)

    assert spans == []


def test_ocr_parser_feeds_into_chunker(tmp_path: Path):
    """PageSpans from OcrPdfParser are accepted by TokenAwareChunker unchanged."""
    pdf_bytes = _make_native_pdf(["First page content.", "Second page content."])
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(pdf_bytes)

    with patch("pytesseract.image_to_string", side_effect=["First page content.", "Second page content."]):
        spans = OcrPdfParser().parse(pdf_path)

    chunker = TokenAwareChunker(chunk_size=64, overlap=8)
    chunks = chunker.chunk(spans, doc_id_prefix="test", title="Test Doc", source="doc.pdf")

    assert len(chunks) >= 1
    # All chunks must have page metadata populated
    for chunk in chunks:
        assert chunk.page_start is not None
        assert chunk.page_end is not None
        assert chunk.source == "doc.pdf"


# ---------------------------------------------------------------------------
# Real Tesseract integration test (skipped when binary absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _TESSERACT_AVAILABLE, reason="Tesseract binary not installed")
def test_ocr_parser_real_tesseract_reads_image_pdf(tmp_path: Path):
    """End-to-end: render a native PDF to images, OCR it, verify text is extracted."""
    native = _make_native_pdf(["Hello OCR world"])
    image_pdf_bytes = _native_pdf_to_image_pdf(native)

    pdf_path = tmp_path / "image_only.pdf"
    pdf_path.write_bytes(image_pdf_bytes)

    spans = OcrPdfParser().parse(pdf_path)
    assert len(spans) >= 1
    combined = " ".join(s.text for s in spans).lower()
    # Tesseract may not be perfect; check at least one keyword
    assert "hello" in combined or "ocr" in combined or "world" in combined
