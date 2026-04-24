"""Async PDF ingestion endpoint.

Accepts a PDF file upload, runs the parse → chunk → BM25-index pipeline in a
FastAPI BackgroundTask, and makes the resulting index immediately available for
retrieval once complete.

Endpoints
---------
POST /v1/ingest          — upload PDF, receive job_id
GET  /v1/ingest/{job_id} — poll job status
"""
from __future__ import annotations

import pickle
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile

from src.api.handlers import get_registry
from src.api.models import IngestJobStatus
from src.ingestion.chunker import TokenAwareChunker, make_doc_id_prefix
from src.ingestion.factory import get_parser
from src.logging_utils import get_logger
from src.retrieval.docstore import load_docstore, save_docstore
from src.retrieval.tokenize import simple_tokenize

logger = get_logger(__name__)

router = APIRouter()

# In-memory job registry: job_id → IngestJobStatus
# Sufficient for MVP; survives the server process lifetime.
_jobs: dict[str, IngestJobStatus] = {}

_DATA_DIR = Path("data/indexes")


def _combine_text(doc) -> str:
    return f"{doc.title} {doc.text}".strip()


def _build_bm25(index_dir: Path, docs: list) -> None:
    from rank_bm25 import BM25Okapi
    tokenized = [simple_tokenize(_combine_text(d)) for d in docs]
    bm25 = BM25Okapi(tokenized)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)


def _run_ingestion(
    job_id: str,
    pdf_path: Path,
    index_dir: Path,
    parser_mode: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    job = _jobs[job_id]

    def _set_progress(p: float) -> None:
        job.progress = p

    try:
        job.status = "processing"
        _set_progress(0.1)

        # 1. Parse
        parser = get_parser(parser_mode)
        pages = parser.parse(pdf_path)
        if not pages:
            raise ValueError(f"No text extracted from '{pdf_path.name}'")
        _set_progress(0.3)

        # 2. Chunk
        chunker = TokenAwareChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        prefix = make_doc_id_prefix(pdf_path.name)
        chunks = chunker.chunk(
            pages,
            doc_id_prefix=prefix,
            title=pdf_path.stem,
            source=pdf_path.name,
        )
        if not chunks:
            raise ValueError("Chunker produced zero chunks")
        _set_progress(0.6)

        # 3. Save docstore
        docstore_path = index_dir / "docstore.jsonl"
        save_docstore(docstore_path, chunks)
        _set_progress(0.8)

        # 4. Build BM25 index
        docs = load_docstore(docstore_path)
        _build_bm25(index_dir, docs)
        _set_progress(0.95)

        # 5. Register with running IndexRegistry
        try:
            get_registry().register(job.index_id)
        except Exception as reg_exc:
            logger.warning("Registry registration failed (non-fatal): %s", reg_exc)

        job.doc_count = len(chunks)
        job.status = "completed"
        job.completed_at = time.time()
        job.progress = 1.0
        logger.info(
            "Ingest job %s completed: %d docs, index_id=%r",
            job_id, len(chunks), job.index_id,
        )

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.completed_at = time.time()
        logger.error("Ingest job %s failed: %s", job_id, exc)


@router.post("/ingest", status_code=202, response_model=IngestJobStatus)
async def create_ingest_job(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    index_id: Annotated[str, Form()],
    parser: Annotated[str, Form()] = "pdf",
    chunk_size: Annotated[int, Form()] = 256,
    chunk_overlap: Annotated[int, Form()] = 32,
) -> IngestJobStatus:
    """Upload a PDF and start an async ingestion job.

    The returned ``job_id`` can be polled at ``GET /v1/ingest/{job_id}``
    until ``status`` is ``completed`` or ``failed``.
    Once completed the index is immediately available via ``POST /v1/retrieve``.
    """
    if parser not in ("pdf", "ocr"):
        raise HTTPException(status_code=422, detail=f"Invalid parser: {parser!r}. Use 'pdf' or 'ocr'.")
    if not index_id.strip():
        raise HTTPException(status_code=422, detail="index_id must not be empty.")

    job_id = str(uuid.uuid4())
    index_dir = _DATA_DIR / index_id
    index_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file synchronously before handing off to background task
    pdf_path = index_dir / (file.filename or "upload.pdf")
    content = await file.read()
    pdf_path.write_bytes(content)

    job = IngestJobStatus(
        job_id=job_id,
        index_id=index_id,
        status="queued",
        created_at=time.time(),
    )
    _jobs[job_id] = job

    background_tasks.add_task(
        _run_ingestion,
        job_id,
        pdf_path,
        index_dir,
        parser,
        chunk_size,
        chunk_overlap,
    )

    logger.info(
        "Ingest job %s queued: index_id=%r, parser=%r, file=%r",
        job_id, index_id, parser, file.filename,
    )
    return job


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
def get_ingest_job(job_id: str) -> IngestJobStatus:
    """Return the current status of an ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job
