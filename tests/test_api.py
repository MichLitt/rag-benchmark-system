"""
A2 API tests.

Uses FastAPI TestClient (httpx) with a toy BM25 index built in tmp_path so no
embedding model is loaded.  Covers:

  - GET /v1/health  (empty registry and after index load)
  - POST /v1/retrieve  (response shape, profile echo-back, score > 0, page metadata)
  - 404 on unknown index
  - top_k truncation
  - Concurrent requests against LazyDocstore (thread-safety regression test)
  - ScoredDocument adapter (_wrap_scores)
  - IndexRegistry: available_index_ids, lazy-load, duplicate-load safety
"""
from __future__ import annotations

import pickle
import threading

import pytest
from fastapi.testclient import TestClient
from rank_bm25 import BM25Okapi

from src.api import handlers
from src.api.index_registry import IndexRegistry
from src.api.server import app
from src.retrieval.docstore import build_docstore_offsets, save_docstore
from src.retrieval.tokenize import simple_tokenize
from src.types import Document, ScoredDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(doc_id="d1", text="Paris is the capital of France.", title="France",
             page_start=1, page_end=1, source="geo.pdf", section="Europe"),
    Document(doc_id="d2", text="London is the capital of England.", title="England",
             page_start=2, page_end=2, source="geo.pdf"),
    Document(doc_id="d3", text="Berlin is the capital of Germany.", title="Germany",
             page_start=3, page_end=3, source="geo.pdf"),
    Document(doc_id="d4", text="Rome is the capital of Italy.", title="Italy",
             page_start=4, page_end=4, source="geo.pdf"),
    Document(doc_id="d5", text="Madrid is the capital of Spain.", title="Spain",
             page_start=5, page_end=5, source="geo.pdf"),
]


def _make_bm25_index(index_dir, docs: list[Document]) -> None:
    """Write docstore.jsonl + docstore.offsets + bm25.pkl into *index_dir*."""
    index_dir.mkdir(parents=True, exist_ok=True)
    save_docstore(index_dir / "docstore.jsonl", docs)
    build_docstore_offsets(index_dir / "docstore.jsonl", index_dir / "docstore.offsets")
    texts = [f"{d.title} {d.text}" for d in docs]
    tokenized = [simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)


@pytest.fixture()
def registry(tmp_path):
    """IndexRegistry pointing at a tmp dir that contains one BM25 index."""
    _make_bm25_index(tmp_path / "geo_idx", SAMPLE_DOCS)
    return IndexRegistry(data_dir=tmp_path), "geo_idx"


@pytest.fixture(autouse=True)
def _reset_registry():
    """Restore the global registry after every test."""
    original = handlers._registry
    yield
    handlers._registry = original


@pytest.fixture()
def client(registry):
    reg, index_id = registry
    handlers.set_registry(reg)
    with TestClient(app) as c:
        yield c, index_id


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------

def test_health_empty_registry(tmp_path):
    handlers.set_registry(IndexRegistry(data_dir=tmp_path))
    with TestClient(app) as c:
        resp = c.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["indexes_loaded"] == []


def test_health_shows_loaded_index(client):
    c, index_id = client
    # Trigger load
    c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": 1})
    resp = c.get("/v1/health")
    assert resp.status_code == 200
    assert index_id in resp.json()["indexes_loaded"]


# ---------------------------------------------------------------------------
# POST /v1/retrieve — response shape
# ---------------------------------------------------------------------------

def test_retrieve_status_ok(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "capital of France", "index_id": index_id})
    assert resp.status_code == 200


def test_retrieve_response_fields(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": 3})
    data = resp.json()
    assert "results" in data
    assert "latency_ms" in data
    assert "retrieval_profile" in data
    assert "index_id" in data


def test_retrieve_result_item_fields(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id})
    for item in resp.json()["results"]:
        assert "doc_id" in item
        assert "text" in item
        assert "score" in item
        assert "metadata" in item
        meta = item["metadata"]
        assert "page_start" in meta
        assert "page_end" in meta
        assert "source" in meta


def test_retrieve_page_metadata_values(client):
    """page_start and page_end must be integers matching the fixture docs."""
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "Paris capital France", "index_id": index_id, "top_k": 1})
    item = resp.json()["results"][0]
    assert item["metadata"]["page_start"] is not None
    assert isinstance(item["metadata"]["page_start"], int)
    assert item["metadata"]["source"] == "geo.pdf"


def test_retrieve_scores_positive(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": 5})
    for item in resp.json()["results"]:
        assert item["score"] > 0


def test_retrieve_score_monotone_decreasing(client):
    """Rank-based scores must be strictly decreasing (1/1 > 1/2 > ...)."""
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": 5})
    scores = [item["score"] for item in resp.json()["results"]]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# POST /v1/retrieve — retrieval_profile echo-back
# ---------------------------------------------------------------------------

def test_retrieve_profile_echoed_default(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "x", "index_id": index_id})
    assert resp.json()["retrieval_profile"] == "auto"


def test_retrieve_profile_echoed_custom(client):
    c, index_id = client
    for profile in ["bm25_v1", "dense_v1", "my_custom_profile_42"]:
        resp = c.post("/v1/retrieve", json={
            "query": "x", "index_id": index_id, "retrieval_profile": profile,
        })
        assert resp.status_code == 200
        assert resp.json()["retrieval_profile"] == profile


def test_retrieve_index_id_echoed(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "x", "index_id": index_id})
    assert resp.json()["index_id"] == index_id


# ---------------------------------------------------------------------------
# POST /v1/retrieve — top_k truncation
# ---------------------------------------------------------------------------

def test_retrieve_top_k_respected(client):
    c, index_id = client
    for k in (1, 2, 3):
        resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": k})
        assert len(resp.json()["results"]) <= k


def test_retrieve_top_k_larger_than_corpus(client):
    """Requesting more results than docs should return at most len(corpus) items."""
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": 100})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) <= len(SAMPLE_DOCS)


# ---------------------------------------------------------------------------
# POST /v1/retrieve — error cases
# ---------------------------------------------------------------------------

def test_retrieve_unknown_index_returns_404(client):
    c, _ = client
    resp = c.post("/v1/retrieve", json={"query": "q", "index_id": "does_not_exist"})
    assert resp.status_code == 404


def test_retrieve_missing_query_returns_422(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"index_id": index_id})
    assert resp.status_code == 422


def test_retrieve_missing_index_id_returns_422(client):
    c, _ = client
    resp = c.post("/v1/retrieve", json={"query": "q"})
    assert resp.status_code == 422


def test_retrieve_invalid_top_k_returns_422(client):
    c, index_id = client
    resp = c.post("/v1/retrieve", json={"query": "q", "index_id": index_id, "top_k": 0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Concurrency: LazyDocstore thread-safety
# ---------------------------------------------------------------------------

def test_concurrent_requests_no_crash(client):
    """Two concurrent retrieve requests must both succeed without docstore errors."""
    c, index_id = client
    results: list[int] = []
    errors: list[str] = []
    lock = threading.Lock()

    def do_request():
        try:
            resp = c.post("/v1/retrieve", json={"query": "capital", "index_id": index_id, "top_k": 3})
            with lock:
                results.append(resp.status_code)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(str(exc))

    threads = [threading.Thread(target=do_request) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent requests raised: {errors}"
    assert all(s == 200 for s in results), f"Status codes: {results}"


def test_lazy_docstore_concurrent_reads(tmp_path):
    """LazyDocstore.get() from 20 threads must return correct, uncorrupted Documents."""
    docs = [
        Document(doc_id=f"d{i}", text=f"document text number {i} " * 5, title=f"Title {i}")
        for i in range(10)
    ]
    ds = tmp_path / "store.jsonl"
    off = tmp_path / "store.offsets"
    save_docstore(ds, docs)
    build_docstore_offsets(ds, off)

    from src.retrieval.docstore import LazyDocstore
    store = LazyDocstore(ds, off)

    retrieved: dict[int, str] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def read(idx: int) -> None:
        try:
            doc = store.get(idx % 10)
            with lock:
                retrieved[idx] = doc.doc_id
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"idx={idx}: {exc}")

    threads = [threading.Thread(target=read, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.close()

    assert errors == [], f"Errors during concurrent reads: {errors}"
    for i, doc_id in retrieved.items():
        assert doc_id == f"d{i % 10}", f"Corrupted doc at thread {i}: got {doc_id}"


# ---------------------------------------------------------------------------
# ScoredDocument adapter (_wrap_scores)
# ---------------------------------------------------------------------------

def test_wrap_scores_basic():
    from src.api.handlers import _wrap_scores
    docs = [Document(doc_id="a", text="hello", title="T")]
    scored = _wrap_scores(docs, [0.9], "bm25")
    assert len(scored) == 1
    assert isinstance(scored[0], ScoredDocument)
    assert scored[0].score == 0.9
    assert scored[0].retrieval_stage == "bm25"
    assert scored[0].document.doc_id == "a"


def test_rank_scores_monotone():
    from src.api.handlers import _rank_scores
    scores = _rank_scores(5)
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# IndexRegistry unit tests
# ---------------------------------------------------------------------------

def test_registry_available_empty(tmp_path):
    reg = IndexRegistry(data_dir=tmp_path)
    assert reg.available_index_ids() == []


def test_registry_available_discovers_index(tmp_path):
    _make_bm25_index(tmp_path / "idx_a", SAMPLE_DOCS)
    reg = IndexRegistry(data_dir=tmp_path)
    assert "idx_a" in reg.available_index_ids()


def test_registry_loaded_empty_before_access(tmp_path):
    _make_bm25_index(tmp_path / "idx_b", SAMPLE_DOCS)
    reg = IndexRegistry(data_dir=tmp_path)
    assert reg.loaded_index_ids() == []


def test_registry_loads_on_first_get(tmp_path):
    _make_bm25_index(tmp_path / "idx_c", SAMPLE_DOCS)
    reg = IndexRegistry(data_dir=tmp_path)
    reg.get_retriever("idx_c")
    assert "idx_c" in reg.loaded_index_ids()


def test_registry_unknown_index_raises(tmp_path):
    reg = IndexRegistry(data_dir=tmp_path)
    with pytest.raises(KeyError):
        reg.get_retriever("ghost_index")


def test_registry_concurrent_load_no_duplicate(tmp_path):
    """Two threads requesting the same index must result in only one load."""
    _make_bm25_index(tmp_path / "shared", SAMPLE_DOCS)
    reg = IndexRegistry(data_dir=tmp_path)
    errors: list[str] = []

    def load():
        try:
            reg.get_retriever("shared")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    threads = [threading.Thread(target=load) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert reg.loaded_index_ids() == ["shared"]
