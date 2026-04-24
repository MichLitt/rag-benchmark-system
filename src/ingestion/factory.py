"""Parser factory: return the appropriate parser for the requested mode."""
from __future__ import annotations

from src.ingestion.ocr_parser import OcrPdfParser
from src.ingestion.pdf_parser import PdfParser


def get_parser(mode: str = "pdf") -> PdfParser | OcrPdfParser:
    """Return a document parser for *mode*.

    Args:
        mode: ``"pdf"`` for native-text PDF parsing (pdfplumber).
              ``"ocr"`` for scanned/image-based PDFs (PyMuPDF + Tesseract).

    Returns:
        A parser instance with a ``parse(path) -> list[PageSpan]`` method.
    """
    if mode == "pdf":
        return PdfParser()
    if mode == "ocr":
        return OcrPdfParser()
    raise ValueError(f"Unknown parser mode: {mode!r}. Valid options: 'pdf', 'ocr'")
