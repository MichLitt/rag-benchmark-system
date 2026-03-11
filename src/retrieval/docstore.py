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
            row = {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}
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

    def _read_offset(self, index: int) -> int:
        if index < 0 or index >= self._num_docs:
            raise IndexError(index)
        self._offsets_file.seek(index * 8)
        raw = self._offsets_file.read(8)
        if len(raw) != 8:
            raise IndexError(index)
        return int(struct.unpack("<Q", raw)[0])

    def get(self, index: int) -> Document:
        offset = self._read_offset(index)
        self._doc_file.seek(offset)
        raw = self._doc_file.readline()
        if not raw:
            raise IndexError(index)
        row = json.loads(raw.decode("utf-8"))
        return Document(
            doc_id=str(row.get("doc_id", "")),
            title=str(row.get("title", "")),
            text=str(row.get("text", "")),
        )
