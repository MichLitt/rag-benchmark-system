from __future__ import annotations

import json
import struct
from pathlib import Path

from src.types import Document


def save_docstore(path: str | Path, docs: list[Document]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for doc in docs:
            row = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "page_start": doc.page_start,
                "page_end": doc.page_end,
                "section": doc.section,
                "source": doc.source,
                "extra_metadata": doc.extra_metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_docstore(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(row.get("doc_id", "")),
                    title=str(row.get("title", "")),
                    text=str(row.get("text", "")),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    section=row.get("section"),
                    source=row.get("source"),
                    extra_metadata=row.get("extra_metadata") or {},
                )
            )
    return docs


def build_docstore_offsets(
    docstore_path: str | Path,
    offsets_path: str | Path,
) -> int:
    src = Path(docstore_path)
    dst = Path(offsets_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with src.open("rb") as doc_file, dst.open("wb") as offsets_file:
        while True:
            offset = doc_file.tell()
            line = doc_file.readline()
            if not line:
                break
            if not line.strip():
                continue
            offsets_file.write(struct.pack("<Q", offset))
            count += 1
    return count


class LazyDocstore:
    """Random-access JSONL docstore backed by a binary offsets sidecar."""

    def __init__(self, path: str | Path, offsets_path: str | Path) -> None:
        self._path = Path(path)
        self._offsets_path = Path(offsets_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Docstore file does not exist: {self._path}")
        if not self._offsets_path.exists():
            raise FileNotFoundError(f"Docstore offsets file does not exist: {self._offsets_path}")

        offsets_size = self._offsets_path.stat().st_size
        if offsets_size % 8 != 0:
            raise ValueError(f"Invalid offsets file size for {self._offsets_path}: {offsets_size}")
        self._num_docs = offsets_size // 8
        self._doc_file = self._path.open("rb")
        self._offsets_file = self._offsets_path.open("rb")

    def __len__(self) -> int:
        return self._num_docs

    def close(self) -> None:
        self._doc_file.close()
        self._offsets_file.close()

    def get(self, index: int) -> Document:
        """Return the Document at *index* using per-request file opens (thread-safe).

        Each call opens both the offsets file and the docstore file independently,
        so concurrent requests from multiple threads never share file-pointer state.
        """
        if index < 0 or index >= self._num_docs:
            raise IndexError(index)
        # Read byte offset from the sidecar (per-request, no shared state)
        with self._offsets_path.open("rb") as off_f:
            off_f.seek(index * 8)
            raw_off = off_f.read(8)
        if len(raw_off) != 8:
            raise IndexError(index)
        offset = int(struct.unpack("<Q", raw_off)[0])
        # Read document row (per-request, no shared state)
        with self._path.open("rb") as doc_f:
            doc_f.seek(offset)
            raw_line = doc_f.readline()
        if not raw_line:
            raise IndexError(index)
        row = json.loads(raw_line.decode("utf-8"))
        return Document(
            doc_id=str(row.get("doc_id", "")),
            title=str(row.get("title", "")),
            text=str(row.get("text", "")),
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            section=row.get("section"),
            source=row.get("source"),
            extra_metadata=row.get("extra_metadata") or {},
        )
