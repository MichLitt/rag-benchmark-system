"""Token-aware sliding-window chunker with page-range and section metadata."""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

import tiktoken

from src.ingestion.pdf_parser import PageSpan
from src.types import Document

logger = logging.getLogger(__name__)

# Heading heuristics: numbered sections, ALL-CAPS short lines, or Title-Cased lines
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+\S"          # 1. / 1.2 / 1.2.3 numbered
    r"|Chapter\s+\d+"                   # Chapter N
    r"|[A-Z][A-Z\s]{3,60}$"            # SHORT ALL-CAPS line
    r")",
    re.MULTILINE,
)


def _detect_section(text: str) -> str | None:
    """Return the first heading-like line found at the start of *text*, or None."""
    first_line = text.lstrip().split("\n", 1)[0].strip()
    if not first_line:
        return None
    if _HEADING_RE.match(first_line) and len(first_line) <= 120:
        return first_line
    return None


class TokenAwareChunker:
    """Sliding-window chunker that counts tokens via tiktoken.

    The chunker operates on a list of :class:`PageSpan` objects.  It
    concatenates all page texts into a flat token stream while keeping a
    parallel per-token page-number map.  A sliding window (``chunk_size``
    tokens, ``overlap`` token stride) then produces :class:`Document` objects
    whose ``page_start`` / ``page_end`` fields reflect the actual pages the
    chunk spans.

    Args:
        chunk_size: Maximum tokens per chunk (default 256).
        overlap: Token overlap between consecutive chunks (default 32).
        encoding_name: tiktoken encoding to use (default ``cl100k_base``).
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        pages: Sequence[PageSpan],
        *,
        doc_id_prefix: str,
        title: str,
        source: str,
    ) -> list[Document]:
        """Chunk *pages* into :class:`Document` objects.

        Args:
            pages: Ordered per-page spans from :class:`PdfParser`.
            doc_id_prefix: Short prefix (e.g. hash of filename) for doc_id generation.
            title: Human-readable title for every chunk (usually the document title).
            source: Origin filename / URL stored in ``Document.source``.

        Returns:
            List of :class:`Document` with ``page_start``, ``page_end``,
            ``section``, and ``source`` populated.
        """
        if not pages:
            return []

        # ---- 1. Build flat token list + per-token page map ----
        all_tokens: list[int] = []
        token_page: list[int] = []          # parallel: page number for each token

        for span in pages:
            page_tokens = self._enc.encode(span.text)
            all_tokens.extend(page_tokens)
            token_page.extend([span.page_num] * len(page_tokens))

        if not all_tokens:
            logger.warning("All pages produced zero tokens — nothing to chunk.")
            return []

        # ---- 2. Sliding window ----
        step = self.chunk_size - self.overlap
        docs: list[Document] = []
        start = 0

        while start < len(all_tokens):
            end = min(start + self.chunk_size, len(all_tokens))
            chunk_tokens = all_tokens[start:end]
            chunk_text = self._enc.decode(chunk_tokens)

            page_start = token_page[start]
            page_end = token_page[end - 1]
            section = _detect_section(chunk_text)

            doc_id = f"{doc_id_prefix}_c{len(docs):04d}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=chunk_text,
                    title=title,
                    page_start=page_start,
                    page_end=page_end,
                    section=section,
                    source=source,
                )
            )

            if end == len(all_tokens):
                break
            start += step

        total_tokens = len(all_tokens)
        logger.info(
            "Chunked %d pages → %d chunks (total %d tokens, chunk_size=%d, overlap=%d)",
            len(pages), len(docs), total_tokens, self.chunk_size, self.overlap,
        )
        return docs


def make_doc_id_prefix(source: str) -> str:
    """Return an 8-hex-char prefix derived from the source filename."""
    return hashlib.sha1(source.encode()).hexdigest()[:8]
