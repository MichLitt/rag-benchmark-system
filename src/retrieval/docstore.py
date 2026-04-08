from __future__ import annotations

import json
import struct
from pathlib import Path

from src.types import Document


def _doc_to_row(doc: Document) -> dict:
    row: dict = {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}
    if doc.page_start is not None:
        row["page_start"] = doc.page_start
    if doc.page_end is not None:
        row["page_end"] = doc.page_end
    if doc.section is not None:
        row["section"] = doc.section
    if doc.source is not None:
        row["source"] = doc.source
    if doc.extra_metadata:
        row["extra_metadata"] = doc.extra_metadata
    return row


def _row_to_doc(row: dict) -> Document:
    return Document(
        doc_id=str(row.get("doc_id", "")),
        title=str(row.get("title", "")),
        text=str(row.get("text", "")),
        page_start=row.get("page_start"),
        page_end=row.get("page_end"),
        section=row.get("section"),
        source=row.get("source"),
        extra_metadata=dict(row.get("extra_metadata") or {}),
    )


def save_docstore(path: str | Path, docs: list[Document]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(_doc_to_row(doc), ensure_ascii=False) + "\n")


def load_docstore(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(_row_to_doc(row))
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
    """Random-access JSONL docstore backed by a binary offsets sidecar.

    Thread-safe: each get() opens fresh file handles per call, avoiding
    shared-handle seek() races under concurrent FastAPI requests.
    """

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

    def __len__(self) -> int:
        return self._num_docs

    def close(self) -> None:
        pass  # No-op — no persistent handles to close

    def _read_offset(self, index: int) -> int:
        if index < 0 or index >= self._num_docs:
            raise IndexError(index)
        with self._offsets_path.open("rb") as f:
            f.seek(index * 8)
            raw = f.read(8)
        if len(raw) != 8:
            raise IndexError(index)
        return int(struct.unpack("<Q", raw)[0])

    def get(self, index: int) -> Document:
        offset = self._read_offset(index)
        with self._path.open("rb") as f:
            f.seek(offset)
            raw = f.readline()
        if not raw:
            raise IndexError(index)
        row = json.loads(raw.decode("utf-8"))
        return _row_to_doc(row)
