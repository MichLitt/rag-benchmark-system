"""Durable ingestion queue backed by SQLite.

The API owns admission and persistence; workers claim jobs independently.  A
lease makes a crashed worker's job eligible for another worker after expiry.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from src.api.models import IngestJobStatus


@dataclass(frozen=True)
class IngestJob:
    job_id: str
    index_id: str
    pdf_path: str
    parser: str
    chunk_size: int
    chunk_overlap: int
    content_sha256: str
    status: str
    progress: float
    doc_count: int | None
    error: str | None
    created_at: float
    started_at: float | None
    completed_at: float | None
    lease_expires_at: float | None
    attempt_count: int


class IngestJobStore:
    """Small SQLite queue with atomic claim/lease/retry semantics."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    parser TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    doc_count INTEGER,
                    error TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    lease_expires_at REAL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(index_id, content_sha256, parser, chunk_size, chunk_overlap)
                )
                """
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> IngestJob:
        return IngestJob(**dict(row))

    @staticmethod
    def _status(job: IngestJob) -> IngestJobStatus:
        return IngestJobStatus(
            job_id=job.job_id,
            index_id=job.index_id,
            status=job.status,
            progress=job.progress,
            doc_count=job.doc_count,
            error=job.error,
            created_at=job.created_at,
            completed_at=job.completed_at,
            attempt_count=job.attempt_count,
        )

    def create_or_get(
        self, *, index_id: str, pdf_path: Path, parser: str, chunk_size: int,
        chunk_overlap: int, content_sha256: str,
    ) -> IngestJobStatus:
        now = time.time()
        with self._connect() as connection:
            existing = connection.execute(
                """SELECT * FROM ingest_jobs WHERE index_id=? AND content_sha256=?
                   AND parser=? AND chunk_size=? AND chunk_overlap=?""",
                (index_id, content_sha256, parser, chunk_size, chunk_overlap),
            ).fetchone()
            if existing is not None:
                return self._status(self._from_row(existing))
            job_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO ingest_jobs (
                    job_id,index_id,pdf_path,parser,chunk_size,chunk_overlap,
                    content_sha256,status,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                (job_id, index_id, str(pdf_path), parser, chunk_size, chunk_overlap,
                 content_sha256, "queued", now),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> IngestJobStatus | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else self._status(self._from_row(row))

    def claim_next(self, *, lease_seconds: float = 60.0) -> IngestJob | None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """UPDATE ingest_jobs SET status='queued', lease_expires_at=NULL,
                   error=COALESCE(error, 'worker lease expired')
                   WHERE status='processing' AND lease_expires_at < ?""",
                (now,),
            )
            row = connection.execute(
                "SELECT * FROM ingest_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job_id = row["job_id"]
            updated = connection.execute(
                """UPDATE ingest_jobs SET status='processing', started_at=COALESCE(started_at, ?),
                   lease_expires_at=?, attempt_count=attempt_count+1, error=NULL
                   WHERE job_id=? AND status='queued'""",
                (now, now + lease_seconds, job_id),
            )
            if updated.rowcount != 1:
                return None
            claimed = connection.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._from_row(claimed)

    def progress(self, job_id: str, value: float, *, lease_seconds: float = 60.0) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE ingest_jobs SET progress=?, lease_expires_at=? WHERE job_id=? AND status='processing'",
                (value, time.time() + lease_seconds, job_id),
            )

    def complete(self, job_id: str, *, doc_count: int) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """UPDATE ingest_jobs SET status='completed', progress=1, doc_count=?,
                   completed_at=?, lease_expires_at=NULL, error=NULL WHERE job_id=?""",
                (doc_count, now, job_id),
            )

    def fail(self, job_id: str, error: str, *, max_attempts: int = 3) -> None:
        now = time.time()
        with self._connect() as connection:
            row = connection.execute("SELECT attempt_count FROM ingest_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return
            terminal = int(row["attempt_count"]) >= max_attempts
            connection.execute(
                """UPDATE ingest_jobs SET status=?, error=?, completed_at=?, lease_expires_at=NULL
                   WHERE job_id=?""",
                ("failed" if terminal else "queued", error, now if terminal else None, job_id),
            )
