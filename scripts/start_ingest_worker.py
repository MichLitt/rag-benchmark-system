"""Run the durable RAG ingestion worker.

The worker shares the index data directory with ``scripts/start_api.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.job_store import IngestJobStore
from src.ingestion.worker import run_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the durable RAG ingestion worker.")
    parser.add_argument("--data-dir", default="data/indexes", help="Shared index root.")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    run_loop(IngestJobStore(data_dir / ".ingest-jobs.sqlite3"), poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
