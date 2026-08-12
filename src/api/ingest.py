"""Async PDF ingestion endpoint.

Accepts a PDF upload and persists a durable ingestion job.  A separate worker
claims the job and runs the parse → chunk → BM25-index pipeline.

Endpoints
---------
POST /v1/ingest          — upload PDF, receive job_id
GET  /v1/ingest/{job_id} — poll job status
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile

from src.api.handlers import get_registry
from src.api.models import IngestJobStatus
from src.ingestion.job_store import IngestJobStore
from src.logging_utils import get_logger

logger = get_logger(__name__)

router = APIRouter()

_INDEX_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DATA_BYTES = 2 * 1024 * 1024 * 1024
MIN_CHUNK_SIZE = 64
MAX_CHUNK_SIZE = 2048


def get_job_store() -> IngestJobStore:
    """Use a queue database adjacent to the configured persistent index root."""
    return IngestJobStore(get_registry().data_dir / ".ingest-jobs.sqlite3")


def _data_usage_bytes(data_dir: Path) -> int:
    if not data_dir.exists():
        return 0
    return sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file())


@router.post("/ingest", status_code=202, response_model=IngestJobStatus)
async def create_ingest_job(
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
    index_id = index_id.strip()
    if not _INDEX_ID_RE.fullmatch(index_id):
        raise HTTPException(
            status_code=422,
            detail=(
                "index_id must be 1-128 characters and contain only letters, "
                "numbers, '.', '_', or '-'."
            ),
        )

    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are accepted.")
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise HTTPException(status_code=422, detail=f"chunk_size must be {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE}.")
    if not 0 <= chunk_overlap < chunk_size:
        raise HTTPException(status_code=422, detail="chunk_overlap must be >= 0 and smaller than chunk_size.")

    # Use the same root configured by scripts/start_api.py --data-dir so a
    # newly ingested index is discoverable by the live registry.
    index_dir = get_registry().data_dir / index_id
    uploads_dir = index_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Save the uploaded file synchronously before the separate worker claims it.
    # UploadFile.filename is client-controlled. Normalize both POSIX and
    # Windows separators before joining it to the configured index directory.
    upload_name = Path((file.filename or "").replace("\\", "/")).name
    if upload_name in {"", ".", ".."}:
        upload_name = "upload.pdf"
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES} byte limit.")
    data_dir = get_registry().data_dir
    if _data_usage_bytes(data_dir) + len(content) > MAX_DATA_BYTES:
        raise HTTPException(status_code=507, detail="Configured ingestion storage quota is exhausted.")
    content_sha256 = hashlib.sha256(content).hexdigest()
    pdf_path = uploads_dir / f"{content_sha256[:16]}-{upload_name}"
    if not pdf_path.exists():
        pdf_path.write_bytes(content)
    job = get_job_store().create_or_get(
        index_id=index_id, pdf_path=pdf_path, parser=parser, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, content_sha256=content_sha256,
    )

    logger.info(
        "Ingest job %s queued: index_id=%r, parser=%r, file=%r",
        job.job_id, index_id, parser, file.filename,
    )
    return job


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
def get_ingest_job(job_id: str) -> IngestJobStatus:
    """Return the current status of an ingestion job."""
    job = get_job_store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    return job
