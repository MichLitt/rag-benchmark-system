from __future__ import annotations

import tiktoken

from src.types import Document


class TokenAwareChunker:
    """Split Documents into token-bounded chunks with overlap.

    Each chunk preserves the page_start/page_end, section, and source from
    the parent Document. A chunk_index is added to extra_metadata so that
    the original sequence can be reconstructed.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"overlap must be in [0, chunk_size), got overlap={overlap} chunk_size={chunk_size}"
            )
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document) -> list[Document]:
        """Split a Document into token-bounded chunks.

        Returns the original document unchanged if its token count is within chunk_size.
        """
        tokens = self._enc.encode(doc.text)
        if len(tokens) <= self._chunk_size:
            return [doc]

        step = self._chunk_size - self._overlap
        chunks: list[Document] = []
        chunk_idx = 0
        start = 0

        while start < len(tokens):
            window = tokens[start : start + self._chunk_size]
            text = self._enc.decode(window)
            chunk_id = f"{doc.doc_id}_c{chunk_idx}"
            chunks.append(Document(
                doc_id=chunk_id,
                text=text,
                title=doc.title,
                page_start=doc.page_start,
                page_end=doc.page_end,
                section=doc.section,
                source=doc.source,
                extra_metadata={**doc.extra_metadata, "chunk_index": chunk_idx},
            ))
            chunk_idx += 1
            start += step
            if start + self._chunk_size > len(tokens) and start < len(tokens):
                # Last partial window — process then stop
                window = tokens[start:]
                text = self._enc.decode(window)
                chunk_id = f"{doc.doc_id}_c{chunk_idx}"
                chunks.append(Document(
                    doc_id=chunk_id,
                    text=text,
                    title=doc.title,
                    page_start=doc.page_start,
                    page_end=doc.page_end,
                    section=doc.section,
                    source=doc.source,
                    extra_metadata={**doc.extra_metadata, "chunk_index": chunk_idx},
                ))
                break

        return chunks
