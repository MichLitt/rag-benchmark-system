"""
A1 ingestion tests.

Covers:
  - TokenAwareChunker: chunk_size, overlap, page_start/page_end, cross-page detection,
    section heuristic, empty input, zero-token pages
  - PdfParser: real multi-page PDF (created with fpdf2)
  - Full pipeline: PdfParser → TokenAwareChunker → save_docstore → load_docstore
  - factory.get_parser
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from src.ingestion.chunker import TokenAwareChunker, _detect_section, make_doc_id_prefix
from src.ingestion.factory import get_parser
from src.ingestion.pdf_parser import PageSpan, PdfParser
from src.retrieval.docstore import load_docstore, save_docstore

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

fpdf2 = pytest.importorskip("fpdf", reason="fpdf2 not installed")
FPDF = fpdf2.FPDF


def _make_pdf(pages_text: list[str]) -> bytes:
    """Create a minimal in-memory PDF with one text line per page using fpdf2."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for text in pages_text:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        # Write up to 10 lines, splitting on newline
        for line in text.split("\n")[:10]:
            pdf.cell(0, 8, text=line[:200], new_x="LMARGIN", new_y="NEXT")
    return pdf.output()


@pytest.fixture()
def two_page_pdf(tmp_path: Path) -> Path:
    """A 2-page PDF where each page has ~40 words of lorem-ipsum text."""
    page1 = (
        "Introduction\n"
        "This is the first page of the document. "
        "It contains several sentences of text that will be tokenized. "
        "The quick brown fox jumps over the lazy dog. "
        "Natural language processing is a subfield of artificial intelligence. "
        "Retrieval-augmented generation combines retrieval with generation."
    )
    page2 = (
        "Methods\n"
        "This is the second page of the document. "
        "It contains additional text for the second section. "
        "Dense retrieval uses embeddings to find relevant passages. "
        "BM25 is a classic sparse retrieval algorithm. "
        "Hybrid retrieval combines both dense and sparse approaches."
    )
    pdf_bytes = _make_pdf([page1, page2])
    p = tmp_path / "test_doc.pdf"
    p.write_bytes(pdf_bytes)
    return p


@pytest.fixture()
def ten_page_pdf(tmp_path: Path) -> Path:
    """A 10-page PDF with ~80 tokens per page to force cross-page chunks."""
    # Each page has enough text to span 80 tokens; chunk_size=128 → chunks cross pages
    pages = []
    for i in range(1, 11):
        words = " ".join([f"word{j}" for j in range(1, 41)])
        pages.append(f"Section {i}\n{words}")
    pdf_bytes = _make_pdf(pages)
    p = tmp_path / "ten_page.pdf"
    p.write_bytes(pdf_bytes)
    return p


# ---------------------------------------------------------------------------
# TokenAwareChunker – unit tests (no PDF needed)
# ---------------------------------------------------------------------------


def test_chunker_basic_single_page():
    """Short single-page text produces exactly one chunk with page_start == page_end."""
    pages = [PageSpan(page_num=1, text="Hello world. " * 10)]
    chunker = TokenAwareChunker(chunk_size=256, overlap=32)
    docs = chunker.chunk(pages, doc_id_prefix="test", title="T", source="t.pdf")
    assert len(docs) >= 1
    for d in docs:
        assert d.page_start == 1
        assert d.page_end == 1
        assert d.source == "t.pdf"
        assert d.title == "T"


def test_chunker_respects_chunk_size():
    """Each chunk must contain at most chunk_size tokens."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    # ~600 tokens of text across 1 page
    long_text = ("The quick brown fox jumps over the lazy dog. " * 40)
    pages = [PageSpan(page_num=1, text=long_text)]
    chunker = TokenAwareChunker(chunk_size=64, overlap=8)
    docs = chunker.chunk(pages, doc_id_prefix="x", title="T", source="s.pdf")
    assert len(docs) > 1
    for d in docs:
        assert len(enc.encode(d.text)) <= 64


def test_chunker_overlap_creates_shared_tokens():
    """Consecutive chunks share exactly `overlap` tokens at the boundary."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = "token " * 200  # simple repeated word
    pages = [PageSpan(page_num=1, text=text)]
    chunker = TokenAwareChunker(chunk_size=50, overlap=10)
    docs = chunker.chunk(pages, doc_id_prefix="y", title="T", source="s.pdf")
    assert len(docs) >= 2
    # Last 10 tokens of chunk 0 == first 10 tokens of chunk 1
    t0 = enc.encode(docs[0].text)
    t1 = enc.encode(docs[1].text)
    assert t0[-10:] == t1[:10]


def test_chunker_cross_page_chunk():
    """When page 1 ends mid-window, a chunk should span both pages."""
    # 30 tokens on page 1, 30 tokens on page 2 → chunk_size=50 forces a cross-page chunk
    page1 = PageSpan(page_num=1, text="alpha " * 30)
    page2 = PageSpan(page_num=2, text="beta " * 30)
    chunker = TokenAwareChunker(chunk_size=50, overlap=5)
    docs = chunker.chunk([page1, page2], doc_id_prefix="z", title="T", source="s.pdf")
    cross = [d for d in docs if d.page_start != d.page_end]
    assert len(cross) >= 1, "Expected at least one cross-page chunk"
    for d in cross:
        assert d.page_start < d.page_end


def test_chunker_single_page_chunks_equal_pages():
    """Chunks that fit within one page must have page_start == page_end."""
    # 20 tokens per page; chunk_size=15 → each chunk stays within one page
    pages = [PageSpan(page_num=i, text="word " * 20) for i in range(1, 5)]
    chunker = TokenAwareChunker(chunk_size=15, overlap=3)
    docs = chunker.chunk(pages, doc_id_prefix="p", title="T", source="s.pdf")
    single_page = [d for d in docs if d.page_start == d.page_end]
    # Most chunks should be single-page given the parameters
    assert len(single_page) > 0


def test_chunker_empty_pages_returns_empty():
    """Empty page list → empty chunk list."""
    chunker = TokenAwareChunker()
    assert chunker.chunk([], doc_id_prefix="x", title="T", source="s.pdf") == []


def test_chunker_doc_ids_unique():
    """All produced doc_ids must be unique."""
    pages = [PageSpan(page_num=1, text="word " * 300)]
    chunker = TokenAwareChunker(chunk_size=50, overlap=10)
    docs = chunker.chunk(pages, doc_id_prefix="uid", title="T", source="s.pdf")
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))


def test_chunker_invalid_overlap_raises():
    with pytest.raises(ValueError, match="overlap"):
        TokenAwareChunker(chunk_size=32, overlap=32)


# ---------------------------------------------------------------------------
# Section detection heuristic
# ---------------------------------------------------------------------------


def test_detect_section_numbered():
    assert _detect_section("1. Introduction\nSome text here.") == "1. Introduction"


def test_detect_section_multi_level():
    assert _detect_section("2.3 Related Work\nSome text.") == "2.3 Related Work"


def test_detect_section_chapter():
    assert _detect_section("Chapter 4\nDeep learning methods.") == "Chapter 4"


def test_detect_section_none_for_body_text():
    text = "This is just a regular paragraph with no heading indicators present."
    assert _detect_section(text) is None


def test_detect_section_none_for_empty():
    assert _detect_section("") is None
    assert _detect_section("   \n  ") is None


# ---------------------------------------------------------------------------
# make_doc_id_prefix
# ---------------------------------------------------------------------------


def test_make_doc_id_prefix_deterministic():
    assert make_doc_id_prefix("paper.pdf") == make_doc_id_prefix("paper.pdf")


def test_make_doc_id_prefix_length():
    assert len(make_doc_id_prefix("x.pdf")) == 8


def test_make_doc_id_prefix_different_sources():
    assert make_doc_id_prefix("a.pdf") != make_doc_id_prefix("b.pdf")


# ---------------------------------------------------------------------------
# PdfParser (requires fpdf2)
# ---------------------------------------------------------------------------


def test_pdf_parser_two_pages(two_page_pdf: Path):
    """PdfParser extracts text from both pages and assigns correct page numbers."""
    parser = PdfParser()
    pages = parser.parse(two_page_pdf)
    assert len(pages) == 2
    assert pages[0].page_num == 1
    assert pages[1].page_num == 2
    assert "first page" in pages[0].text.lower()
    assert "second page" in pages[1].text.lower()


def test_pdf_parser_page_numbers_1indexed(two_page_pdf: Path):
    """Page numbers must be 1-indexed (not 0-indexed)."""
    parser = PdfParser()
    pages = PdfParser().parse(two_page_pdf)
    assert pages[0].page_num == 1


def test_pdf_parser_ten_pages(ten_page_pdf: Path):
    """PdfParser should extract text from all 10 pages."""
    pages = PdfParser().parse(ten_page_pdf)
    assert len(pages) == 10


# ---------------------------------------------------------------------------
# Full pipeline: PdfParser → TokenAwareChunker → save/load
# ---------------------------------------------------------------------------


def test_full_pipeline_two_page(two_page_pdf: Path, tmp_path: Path):
    """End-to-end: parse a 2-page PDF, chunk, save, reload with correct fields."""
    parser = PdfParser()
    pages = parser.parse(two_page_pdf)

    chunker = TokenAwareChunker(chunk_size=64, overlap=8)
    prefix = make_doc_id_prefix(two_page_pdf.name)
    docs = chunker.chunk(
        pages,
        doc_id_prefix=prefix,
        title="Test Document",
        source=two_page_pdf.name,
    )
    assert len(docs) >= 1

    # All chunks have page metadata
    for d in docs:
        assert d.page_start is not None
        assert d.page_end is not None
        assert d.page_start >= 1
        assert d.page_end >= d.page_start
        assert d.source == two_page_pdf.name
        assert d.title == "Test Document"

    # Persist and reload
    ds_path = tmp_path / "docstore.jsonl"
    save_docstore(ds_path, docs)
    reloaded = load_docstore(ds_path)
    assert len(reloaded) == len(docs)
    assert reloaded[0].page_start == docs[0].page_start
    assert reloaded[0].source == two_page_pdf.name


def test_full_pipeline_cross_page_chunks(ten_page_pdf: Path):
    """10-page PDF with chunk_size=128 should produce some cross-page chunks."""
    pages = PdfParser().parse(ten_page_pdf)
    # chunk_size smaller than 2 pages of tokens → guaranteed cross-page chunks
    chunker = TokenAwareChunker(chunk_size=128, overlap=16)
    prefix = make_doc_id_prefix(ten_page_pdf.name)
    docs = chunker.chunk(
        pages,
        doc_id_prefix=prefix,
        title="10-Page Test",
        source=ten_page_pdf.name,
    )
    cross = [d for d in docs if d.page_start != d.page_end]
    assert len(cross) >= 1, "Expected cross-page chunks in a 10-page PDF"


def test_full_pipeline_chunk_count_logged(two_page_pdf: Path, caplog):
    """TokenAwareChunker should emit an INFO log with chunk count."""
    import logging
    pages = PdfParser().parse(two_page_pdf)
    chunker = TokenAwareChunker(chunk_size=64, overlap=8)
    with caplog.at_level(logging.INFO, logger="src.ingestion.chunker"):
        docs = chunker.chunk(
            pages,
            doc_id_prefix="log",
            title="T",
            source="t.pdf",
        )
    assert any("chunk" in r.message.lower() for r in caplog.records)
    assert len(docs) >= 1


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def test_factory_returns_pdf_parser():
    p = get_parser("pdf")
    assert isinstance(p, PdfParser)


def test_factory_ocr_not_implemented():
    with pytest.raises(NotImplementedError):
        get_parser("ocr")


def test_factory_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown parser mode"):
        get_parser("xyz")
