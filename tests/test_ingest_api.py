"""Tests for POST /v1/ingest and GET /v1/ingest/{job_id}.

Uses FastAPI TestClient with a real (small) native-text PDF so the full
parse → chunk → BM25 → register pipeline runs in-process.

The API only enqueues. Tests explicitly execute one independent worker turn,
which also proves that API and worker communicate through durable storage.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import handlers, ingest
from src.api.index_registry import IndexRegistry
from src.api.server import app
from src.ingestion.worker import process_one

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
    yield


def _process_queued_job() -> None:
    assert process_one(ingest.get_job_store())


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
        _process_queued_job()
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
        _process_queued_job()
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
        _process_queued_job()
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


def test_ingest_rejects_index_id_path_traversal():
    pdf_bytes = _make_pdf_bytes(["text"])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/ingest",
            data={"index_id": "../../outside"},
            files={"file": ("f.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert resp.status_code == 422


def test_ingest_sanitizes_uploaded_filename(tmp_path: Path):
    pdf_bytes = _make_pdf_bytes(["safe upload"])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/ingest",
            data={"index_id": "safe-index"},
            files={
                "file": (
                    "../../outside.pdf",
                    io.BytesIO(pdf_bytes),
                    "application/pdf",
                )
            },
        )

    assert resp.status_code == 202
    assert (tmp_path / "indexes" / "safe-index" / "uploads" / "outside.pdf").exists() is False
    assert list((tmp_path / "indexes" / "safe-index" / "uploads").glob("*-outside.pdf"))
    assert not (tmp_path / "outside.pdf").exists()


def test_get_unknown_job_returns_404():
    with TestClient(app) as client:
        resp = client.get("/v1/ingest/nonexistent-job-id")
    assert resp.status_code == 404


def test_ingest_deduplicates_same_content_and_config():
    pdf_bytes = _make_pdf_bytes(["idempotent upload"])
    with TestClient(app) as client:
        first = client.post(
            "/v1/ingest", data={"index_id": "dedupe"},
            files={"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        second = client.post(
            "/v1/ingest", data={"index_id": "dedupe"},
            files={"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert first.status_code == second.status_code == 202
    assert first.json()["job_id"] == second.json()["job_id"]


def test_expired_worker_lease_is_reclaimed():
    pdf_bytes = _make_pdf_bytes(["recover after worker crash"])
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest", data={"index_id": "reclaim"},
            files={"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        store = ingest.get_job_store()
        claimed = store.claim_next(lease_seconds=-1)
        assert claimed is not None
        _process_queued_job()
        job = client.get(f"/v1/ingest/{response.json()['job_id']}").json()
    assert job["status"] == "completed"
    assert job["attempt_count"] == 2


def test_queued_job_survives_api_restart_and_is_completed(tmp_path: Path):
    """A new API registry/process can read the durable queue left by its predecessor."""
    pdf_bytes = _make_pdf_bytes(["durable job across an API restart"])
    with TestClient(app) as first_api:
        created = first_api.post(
            "/v1/ingest", data={"index_id": "api-restart"},
            files={"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert created.status_code == 202
        job_id = created.json()["job_id"]

    # Simulate a fresh API process: it receives a new registry but shares the
    # same data directory and thus the same SQLite queue database.
    handlers.set_registry(IndexRegistry(data_dir=tmp_path / "indexes"))
    with TestClient(app) as restarted_api:
        queued = restarted_api.get(f"/v1/ingest/{job_id}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        _process_queued_job()
        completed = restarted_api.get(f"/v1/ingest/{job_id}").json()

    assert completed["status"] == "completed"
    assert completed["doc_count"] > 0


def test_ingest_rejects_wrong_content_type_and_invalid_chunking():
    with TestClient(app) as client:
        wrong_type = client.post(
            "/v1/ingest", data={"index_id": "wrongtype"},
            files={"file": ("manual.txt", io.BytesIO(b"not a PDF"), "text/plain")},
        )
        invalid_chunk = client.post(
            "/v1/ingest", data={"index_id": "badchunk", "chunk_size": "32"},
            files={"file": ("manual.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
    assert wrong_type.status_code == 415
    assert invalid_chunk.status_code == 422


def test_ingest_rejects_upload_when_storage_quota_is_exhausted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ingest, "MAX_DATA_BYTES", 1)
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest", data={"index_id": "quota"},
            files={"file": ("manual.pdf", io.BytesIO(b"%PDF-data"), "application/pdf")},
        )
    assert response.status_code == 507


def test_worker_retries_failure_then_marks_job_terminal():
    with TestClient(app) as client:
        response = client.post(
            "/v1/ingest", data={"index_id": "retry"},
            files={"file": ("broken.pdf", io.BytesIO(b"not a real PDF"), "application/pdf")},
        )
        assert response.status_code == 202
        for _ in range(3):
            _process_queued_job()
        job = client.get(f"/v1/ingest/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["attempt_count"] == 3
    assert job["error"]


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
