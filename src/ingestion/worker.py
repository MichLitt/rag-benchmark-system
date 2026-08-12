"""Independent worker for durable PDF ingestion jobs."""
from __future__ import annotations

import pickle
import time
from pathlib import Path

from src.ingestion.chunker import TokenAwareChunker, make_doc_id_prefix
from src.ingestion.factory import get_parser
from src.ingestion.job_store import IngestJob, IngestJobStore
from src.logging_utils import get_logger
from src.retrieval.docstore import load_docstore, save_docstore
from src.retrieval.tokenize import simple_tokenize

logger = get_logger(__name__)


def _build_bm25(index_dir: Path, docs: list) -> None:
    from rank_bm25 import BM25Okapi

    tokenized = [simple_tokenize(f"{doc.title} {doc.text}".strip()) for doc in docs]
    with open(index_dir / "bm25.pkl", "wb") as handle:
        pickle.dump(BM25Okapi(tokenized), handle)


def process_one(store: IngestJobStore, *, lease_seconds: float = 60.0) -> bool:
    """Claim and execute one job. Returns ``True`` only when a job was claimed."""
    job = store.claim_next(lease_seconds=lease_seconds)
    if job is None:
        return False
    try:
        _process_claimed(store, job, lease_seconds=lease_seconds)
    except Exception as exc:  # noqa: BLE001 - worker must persist all failures
        store.fail(job.job_id, str(exc))
        logger.exception("Ingest job %s failed", job.job_id)
    return True


def _process_claimed(store: IngestJobStore, job: IngestJob, *, lease_seconds: float) -> None:
    pdf_path = Path(job.pdf_path)
    index_dir = pdf_path.parents[1]
    store.progress(job.job_id, 0.1, lease_seconds=lease_seconds)

    pages = get_parser(job.parser).parse(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from '{pdf_path.name}'")
    store.progress(job.job_id, 0.3, lease_seconds=lease_seconds)

    chunks = TokenAwareChunker(chunk_size=job.chunk_size, overlap=job.chunk_overlap).chunk(
        pages,
        doc_id_prefix=make_doc_id_prefix(pdf_path.name),
        title=pdf_path.stem,
        source=pdf_path.name,
    )
    if not chunks:
        raise ValueError("Chunker produced zero chunks")
    store.progress(job.job_id, 0.6, lease_seconds=lease_seconds)

    docstore_path = index_dir / "docstore.jsonl"
    save_docstore(docstore_path, chunks)
    store.progress(job.job_id, 0.8, lease_seconds=lease_seconds)
    _build_bm25(index_dir, load_docstore(docstore_path))
    # This eagerly refreshes a colocated API process. A separately deployed API
    # still discovers the completed index lazily from the shared data volume.
    try:
        from src.api.handlers import get_registry

        get_registry().register(job.index_id)
    except Exception as exc:  # noqa: BLE001 - registration is an optimization
        logger.warning("Index registration deferred for %s: %s", job.index_id, exc)
    store.complete(job.job_id, doc_count=len(chunks))
    logger.info("Ingest job %s completed: %d docs", job.job_id, len(chunks))


def run_loop(store: IngestJobStore, *, poll_seconds: float = 1.0) -> None:
    """Continuously process jobs; intentionally has no API-process dependency."""
    while True:
        if not process_one(store):
            time.sleep(poll_seconds)
