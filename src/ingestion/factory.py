from __future__ import annotations

from src.ingestion.pdf_parser import PdfParser


def get_parser(mode: str) -> PdfParser:
    """Return an ingestion parser for the given mode.

    Args:
        mode: Parser mode. Currently only "pdf" is supported.

    Returns:
        A parser instance.

    Raises:
        ValueError: If mode is not supported.
    """
    if mode == "pdf":
        return PdfParser()
    raise ValueError(f"Unknown parser mode: {mode!r}. Supported modes: 'pdf'")
