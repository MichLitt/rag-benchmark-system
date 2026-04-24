"""Tests for POST /v1/ingest and GET /v1/ingest/{job_id}.

Uses FastAPI TestClient with a real (small) native-text PDF so the full
parse → chunk → BM25 → register pipeline runs in-process.

The background task executes synchronously inside TestClient because
FastAPI TestClient processes background tasks before returning in test mode.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import handlers, ingest
from src.api.index_registry import IndexRegistry
from src.api.server import app

fpdf2 = pytest.importorskip("fpdf", reason="fpdf2 not installed")
FPDF = fpdf2.FPDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_bytes(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in text.split("\n")[:10]:
            pdf.cell(0, 8, text=line[:200], new_x="LMARGIN", new_y="NEXT")
    return pdf.output()


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Give each test its own IndexRegistry and job store pointing at tmp_path."""
    reg = IndexRegistry(data_dir=tmp_path / "indexes")
    monkeypatch.setattr(handlers, "_registry", reg)
    monkeypatch.setattr(ingest, "_DATA_DIR", tmp_path / "indexes")
    ingest._jobs.clear()
    yield
    ingest._jobs.clear()


# ---------------------------------------------------------------------------
# POST /v1/ingest
# ---------------------------------------------------------------------------

def test_ingest_returns_202_and_job_id():
    pdf_bytes = _make_pdf_bytes(["Hello world document."])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/ingest",
            data={"index_id": "testidx"},
            files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["index_id"] == "testidx"
    assert body["status"] in ("queued", "processing", "completed")


def test_ingest_job_completes(tmp_path: Path):
    pdf_bytes = _make_pdf_bytes([
        "Introduction\nThis document discusses retrieval-augmented generation.",
        "Methods\nDense retrieval uses FAISS and BM25 is a sparse baseline.",
    ])
    with TestClient(app) as client:
        post_resp = client.post(
            "/v1/ingest",
            data={"index_id": "myidx"},
            files={"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]

        get_resp = client.get(f"/v1/ingest/{job_id}")
    assert get_resp.status_code == 200
    job = get_resp.json()
    assert job["status"] == "completed", f"Unexpected status: {job}"
    assert job["doc_count"] is not None and job["doc_count"] > 0
    assert job["completed_at"] is not None


def test_ingest_index_appears_in_health():
    pdf_bytes = _make_pdf_bytes(["Some content to index."])
    with TestClient(app) as client:
        client.post(
            "/v1/ingest",
            data={"index_id": "healthidx"},
            files={"file": ("f.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        health = client.get("/v1/health").json()
    assert "healthidx" in health["indexes_loaded"]


def test_ingest_index_is_retrievable():
    pdf_bytes = _make_pdf_bytes([
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence and machine learning are transforming industries."
    ])
    with TestClient(app) as client:
        client.post(
            "/v1/ingest",
            data={"index_id": "retrieveidx"},
            files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        retrieve_resp = client.post(
            "/v1/retrieve",
            json={"query": "artificial intelligence", "index_id": "retrieveidx", "top_k": 3},
        )
    assert retrieve_resp.status_code == 200
    results = retrieve_resp.json()["results"]
    assert len(results) > 0
    assert all("text" in r for r in results)


def test_ingest_invalid_parser_returns_422():
    pdf_bytes = _make_pdf_bytes(["text"])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/ingest",
            data={"index_id": "idx", "parser": "docx"},
            files={"file": ("f.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert resp.status_code == 422


def test_ingest_empty_index_id_returns_422():
    pdf_bytes = _make_pdf_bytes(["text"])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/ingest",
            data={"index_id": "   "},
            files={"file": ("f.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert resp.status_code == 422


def test_get_unknown_job_returns_404():
    with TestClient(app) as client:
        resp = client.get("/v1/ingest/nonexistent-job-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# IndexRegistry.register()
# ---------------------------------------------------------------------------

def test_registry_register_makes_index_available(tmp_path: Path):
    """register() loads a freshly-written BM25 index into the registry."""
    import pickle
    from rank_bm25 import BM25Okapi

    from src.api.index_registry import IndexRegistry
    from src.retrieval.docstore import save_docstore
    from src.retrieval.tokenize import simple_tokenize
    from src.types import Document

    docs = [Document(doc_id="d1", text="hello world", title="T")]
    index_dir = tmp_path / "indexes" / "myidx"
    index_dir.mkdir(parents=True)
    save_docstore(index_dir / "docstore.jsonl", docs)
    bm25 = BM25Okapi([simple_tokenize(d.text) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    reg = IndexRegistry(data_dir=tmp_path / "indexes")
    assert "myidx" not in reg.loaded_index_ids()
    reg.register("myidx")
    assert "myidx" in reg.loaded_index_ids()


def test_registry_register_is_idempotent(tmp_path: Path):
    """Calling register() twice on the same index is a no-op."""
    import pickle
    from rank_bm25 import BM25Okapi

    from src.api.index_registry import IndexRegistry
    from src.retrieval.docstore import save_docstore
    from src.retrieval.tokenize import simple_tokenize
    from src.types import Document

    docs = [Document(doc_id="d1", text="hello world", title="T")]
    index_dir = tmp_path / "indexes" / "dupidx"
    index_dir.mkdir(parents=True)
    save_docstore(index_dir / "docstore.jsonl", docs)
    bm25 = BM25Okapi([simple_tokenize(d.text) for d in docs])
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    reg = IndexRegistry(data_dir=tmp_path / "indexes")
    reg.register("dupidx")
    reg.register("dupidx")  # second call must not raise
    assert reg.loaded_index_ids().count("dupidx") == 1
