"""FastAPI application: route registration and ASGI entry-point.

Start with:
    uv run python scripts/start_api.py
or directly:
    uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from src.api.handlers import handle_health, handle_retrieve
from src.api.models import HealthResponse, RetrieveRequest, RetrieveResponse

app = FastAPI(
    title="RAG Retrieval API",
    description="Read-only retrieval service for pre-built FAISS / BM25 indexes.",
    version="0.1.0",
)


@app.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Search a pre-built index and return top-k documents with scores."""
    return await run_in_threadpool(handle_retrieve, req)


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service status and the list of in-memory index IDs."""
    return handle_health()
