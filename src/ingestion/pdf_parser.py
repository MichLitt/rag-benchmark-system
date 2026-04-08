from __future__ import annotations

from pathlib import Path

from src.types import Document


class PdfParser:
    """Parse a native-text PDF into a list of Documents, one per page.

    Uses pdfplumber for text extraction. Scanned/image-only PDFs are not
    supported (OCR is out of scope for Phase 2 MVP).
    """

    def parse(
        self,
        pdf_path: str | Path,
        *,
        source: str | None = None,
    ) -> list[Document]:
        """Parse a PDF file and return one Document per non-empty page.

        Args:
            pdf_path: Path to the PDF file.
            source: Optional source label stored in Document.source.
                    Defaults to the file path string.

        Returns:
            List of Documents with page_start == page_end == page number (1-indexed).
            Pages with no extractable text are silently skipped.
        """
        import pdfplumber  # lazy import — optional dependency

        path = Path(pdf_path)
        src = source or str(path)
        docs: list[Document] = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue
                doc_id = f"{path.stem}_p{page_num}"
                docs.append(Document(
                    doc_id=doc_id,
                    text=text,
                    title=path.stem,
                    page_start=page_num,
                    page_end=page_num,
                    source=src,
                ))

        return docs
