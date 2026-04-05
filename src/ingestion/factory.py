"""Parser factory: return the appropriate parser for the requested mode."""
from __future__ import annotations

from src.ingestion.pdf_parser import PdfParser


def get_parser(mode: str = "pdf") -> PdfParser:
    """Return a document parser for *mode*.

    Args:
        mode: ``"pdf"`` for native-text PDF parsing (pdfplumber).
              ``"ocr"`` is a stretch goal and raises ``NotImplementedError``.

    Returns:
        A parser instance with a ``parse(path) -> list[PageSpan]`` method.
    """
    if mode == "pdf":
        return PdfParser()
    if mode == "ocr":
        raise NotImplementedError(
            "OCR parser is a stretch goal and has not been implemented yet."
        )
    raise ValueError(f"Unknown parser mode: {mode!r}. Valid options: 'pdf'")
