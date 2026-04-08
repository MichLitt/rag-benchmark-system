"""PDF parsing layer: extract per-page text using pdfplumber (native-text PDFs only)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class PageSpan:
    """Text content extracted from a single PDF page."""
    page_num: int   # 1-indexed
    text: str


class PdfParser:
    """Parse a native-text PDF into a list of per-page PageSpan objects.

    Each PageSpan contains the full text of one page and its 1-indexed page number.
    Empty pages (no extractable text) are skipped with a debug-level log.

    OCR/scanned-PDF support is a stretch goal not implemented here.
    """

    def parse(self, path: str | Path) -> list[PageSpan]:
        """Return one PageSpan per non-empty page in the PDF.

        Args:
            path: Path to a native-text PDF file.

        Returns:
            List of PageSpan, one entry per page that yielded extractable text.
        """
        path = Path(path)
        pages: list[PageSpan] = []
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(PageSpan(page_num=i, text=text))
                else:
                    logger.debug("Page %d/%d yielded no text — skipped", i, total)
        logger.info(
            "Parsed %d/%d pages with text from '%s'",
            len(pages), total, path.name,
        )
        return pages
