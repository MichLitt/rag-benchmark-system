"""
A0 migration tests: verify that old 3-field JSONL format is still loadable
and that new fields round-trip correctly through save/load and LazyDocstore.
"""
from __future__ import annotations

import json

import pytest

from src.retrieval.docstore import (
    LazyDocstore,
    build_docstore_offsets,
    load_docstore,
    save_docstore,
)
from src.types import Document


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

OLD_ROWS = [
    {"doc_id": "d1", "title": "First Title", "text": "First passage text."},
    {"doc_id": "d2", "title": "Second Title", "text": "Second passage text."},
]


def _write_old_format(path, rows):
    """Write JSONL with only the legacy 3-field schema (doc_id/title/text)."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# load_docstore – backward compat
# ---------------------------------------------------------------------------


def test_load_old_format_backward_compat(tmp_path):
    """load_docstore must succeed on old 3-field JSONL; new fields default to None/{}."""
    ds_path = tmp_path / "old.jsonl"
    _write_old_format(ds_path, OLD_ROWS)

    docs = load_docstore(ds_path)
    assert len(docs) == 2
    d = docs[0]
    assert d.doc_id == "d1"
    assert d.title == "First Title"
    assert d.text == "First passage text."
    assert d.page_start is None
    assert d.page_end is None
    assert d.section is None
    assert d.source is None
    assert d.extra_metadata == {}


# ---------------------------------------------------------------------------
# save_docstore + load_docstore – new format round-trip
# ---------------------------------------------------------------------------


def test_save_and_reload_new_format(tmp_path):
    """save_docstore writes all fields; reload produces identical Documents."""
    doc = Document(
        doc_id="p1",
        text="Some chunk text.",
        title="My Doc",
        page_start=3,
        page_end=4,
        section="Introduction",
        source="my_doc.pdf",
        extra_metadata={"author": "Alice"},
    )
    ds_path = tmp_path / "new.jsonl"
    save_docstore(ds_path, [doc])

    reloaded = load_docstore(ds_path)
    assert len(reloaded) == 1
    r = reloaded[0]
    assert r.doc_id == "p1"
    assert r.page_start == 3
    assert r.page_end == 4
    assert r.section == "Introduction"
    assert r.source == "my_doc.pdf"
    assert r.extra_metadata == {"author": "Alice"}


def test_single_page_chunk_round_trips(tmp_path):
    """Single-page chunk (page_start == page_end) survives serialization."""
    doc = Document(doc_id="x", text="t", title="T", page_start=5, page_end=5)
    ds_path = tmp_path / "sp.jsonl"
    save_docstore(ds_path, [doc])
    r = load_docstore(ds_path)[0]
    assert r.page_start == 5
    assert r.page_end == 5


def test_cross_page_chunk_round_trips(tmp_path):
    """Cross-page chunk (page_start != page_end) survives serialization."""
    doc = Document(doc_id="y", text="t", title="T", page_start=3, page_end=4)
    ds_path = tmp_path / "cp.jsonl"
    save_docstore(ds_path, [doc])
    r = load_docstore(ds_path)[0]
    assert r.page_start == 3
    assert r.page_end == 4


# ---------------------------------------------------------------------------
# LazyDocstore – backward compat
# ---------------------------------------------------------------------------


def test_lazy_docstore_old_format(tmp_path):
    """LazyDocstore must load old 3-field JSONL with new fields defaulting to None."""
    ds_path = tmp_path / "old.jsonl"
    offsets_path = tmp_path / "old.offsets"
    _write_old_format(ds_path, OLD_ROWS)
    build_docstore_offsets(ds_path, offsets_path)

    store = LazyDocstore(ds_path, offsets_path)
    try:
        assert len(store) == 2
        doc = store.get(0)
        assert doc.doc_id == "d1"
        assert doc.page_start is None
        assert doc.page_end is None
        assert doc.extra_metadata == {}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# LazyDocstore – new format
# ---------------------------------------------------------------------------


def test_lazy_docstore_new_format(tmp_path):
    """LazyDocstore must correctly load all new fields from new-format JSONL."""
    doc = Document(
        doc_id="p2",
        text="Chunk text.",
        title="PDF Doc",
        page_start=7,
        page_end=8,
        section="Methods",
        source="paper.pdf",
        extra_metadata={"year": 2024},
    )
    ds_path = tmp_path / "new.jsonl"
    offsets_path = tmp_path / "new.offsets"
    save_docstore(ds_path, [doc])
    build_docstore_offsets(ds_path, offsets_path)

    store = LazyDocstore(ds_path, offsets_path)
    try:
        r = store.get(0)
        assert r.page_start == 7
        assert r.page_end == 8
        assert r.section == "Methods"
        assert r.source == "paper.pdf"
        assert r.extra_metadata == {"year": 2024}
    finally:
        store.close()
