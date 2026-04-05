"""Launch the RAG Retrieval API with uvicorn.

Usage:
    uv run python scripts/start_api.py [--host HOST] [--port PORT] [--reload]
    uv run python scripts/start_api.py --data-dir data/indexes --port 8080

The server exposes:
    POST /v1/retrieve   — query a pre-built index
    GET  /v1/health     — liveness check + loaded index list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start the RAG Retrieval API server.")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
    p.add_argument("--port", type=int, default=8080, help="Bind port (default 8080).")
    p.add_argument("--data-dir", default="data/indexes", help="Index root directory.")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload.")
    p.add_argument("--log-level", default="info", help="Uvicorn log level.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Configure the registry before importing the app so the data_dir is set
    from src.api.handlers import set_registry
    from src.api.index_registry import IndexRegistry

    set_registry(IndexRegistry(data_dir=args.data_dir))

    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
