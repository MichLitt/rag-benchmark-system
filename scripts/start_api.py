#!/usr/bin/env python3
"""Start the FastAPI retrieval service with uvicorn.

Usage:
    uv run python scripts/start_api.py --port 8080

    # With a pre-built index loaded at startup:
    INDEX_CONFIG_DEFAULT=config/wiki18_21m_sharded.yaml \\
        uv run python scripts/start_api.py --port 8080

    # Then query:
    curl -X POST http://localhost:8080/v1/retrieve \\
        -H "Content-Type: application/json" \\
        -d '{"query": "who founded Apple", "top_k": 5}'

Note: Keep --workers 1. FAISS indexes loaded into memory are not
fork-safe; multiple workers would each need to reload the index.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch the Agent Knowledge Retrieval API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host.")
    p.add_argument("--port", type=int, default=8080, help="Bind port.")
    p.add_argument(
        "--workers", type=int, default=1,
        help="Uvicorn worker count. Keep at 1 — FAISS indexes are not fork-safe.",
    )
    p.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload for development (forces workers=1).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    workers = 1 if args.reload else args.workers
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        workers=workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
