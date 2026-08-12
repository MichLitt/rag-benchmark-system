"""OCR parsing layer: extract text from scanned/image-based PDFs.

Uses PyMuPDF (fitz) to render each page to a high-resolution image, then
Tesseract (via pytesseract) to extract text from the rendered image.

This handles documents where pdfplumber yields no extractable text because
the content is stored as images rather than native text.

Requirements (system):
    - Tesseract OCR binary must be installed:
        macOS:  brew install tesseract
        Linux:  apt-get install tesseract-ocr
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from src.ingestion.pdf_parser import PageSpan

logger = logging.getLogger(__name__)

# Render resolution multiplier. 2× gives ~144 DPI from a 72 DPI PDF baseline,
# which provides acceptable OCR accuracy without excessive memory usage.
_ZOOM = 2


class OcrPdfParser:
    """Parse a scanned or image-based PDF into per-page PageSpan objects.

    Uses PyMuPDF for page rendering and pytesseract for OCR. The interface is
    identical to PdfParser so it slots in without changes to downstream code.

    Empty pages (no text after stripping whitespace) are skipped.
    """

    def parse(self, path: str | Path) -> list[PageSpan]:
        """Return one PageSpan per non-empty page after OCR.

        Args:
            path: Path to a PDF file (scanned or image-based).

        Returns:
            List of PageSpan, one entry per page that yielded extractable text.
        """
        import fitz  # PyMuPDF — imported lazily to avoid hard startup cost
        import pytesseract
        from PIL import Image

        path = Path(path)
        pages: list[PageSpan] = []
        matrix = fitz.Matrix(_ZOOM, _ZOOM)

        doc = fitz.open(str(path))
        total = len(doc)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text: str = pytesseract.image_to_string(img)
            if text.strip():
                pages.append(PageSpan(page_num=i, text=text))
            else:
                logger.debug("Page %d/%d yielded no OCR text — skipped", i, total)
        doc.close()

        logger.info(
            "OCR parsed %d/%d pages with text from '%s'",
            len(pages), total, path.name,
        )
        return pages
