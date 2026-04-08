from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

from src.types import Document


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    mode = "rt"
    with opener(file_path, mode, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _split_contents(contents: str) -> tuple[str, str]:
    if "[SEP]" not in contents:
        raw = contents.strip()
        if "\n" in raw:
            maybe_title, maybe_text = raw.split("\n", 1)
            title = maybe_title.strip().strip('"').strip()
            text = maybe_text.strip()
            if title and text:
                return title, text
        return "", raw
    title, text = contents.split("[SEP]", 1)
    return title.strip(), text.strip()


def _row_to_document(row: dict) -> Document:
    doc_id = str(row.get("doc_id") or row.get("id") or "")
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    contents = str(row.get("contents") or "").strip()

    if contents and (not text or "[SEP]" in contents):
        parsed_title, parsed_text = _split_contents(contents)
        if parsed_title:
            title = parsed_title
        if parsed_text:
            text = parsed_text

    return Document(
        doc_id=doc_id,
        title=title,
        text=text,
        page_start=int(row["page_start"]) if row.get("page_start") is not None else None,
        page_end=int(row["page_end"]) if row.get("page_end") is not None else None,
        section=str(row["section"]) if row.get("section") is not None else None,
        source=str(row["source"]) if row.get("source") is not None else None,
        extra_metadata=dict(row.get("extra_metadata") or {}),
    )


def iter_corpus_documents(path: str | Path) -> Iterator[Document]:
    for row in iter_jsonl(path):
        yield _row_to_document(row)


def load_documents(path: str | Path, max_docs: int | None = None) -> list[Document]:
    docs: list[Document] = []
    for doc in iter_corpus_documents(path):
        docs.append(doc)
        if max_docs is not None and max_docs > 0 and len(docs) >= max_docs:
            break
    return docs
