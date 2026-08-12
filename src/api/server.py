"""FastAPI application: route registration and ASGI entry-point.

Start with:
    uv run python scripts/start_api.py
or directly:
    uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool

from src.api.handlers import handle_health, handle_retrieve
from src.api.ingest import router as ingest_router
from src.api.auth import require_api_token
from src.api.models import HealthResponse, RetrieveRequest, RetrieveResponse

app = FastAPI(
    title="RAG Retrieval API",
    description="Retrieval service for pre-built FAISS / BM25 indexes with async PDF ingestion.",
    version="0.2.0",
)

app.include_router(ingest_router, prefix="/v1", dependencies=[Depends(require_api_token)])


@app.post("/v1/retrieve", response_model=RetrieveResponse, dependencies=[Depends(require_api_token)])
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Search a pre-built index and return top-k documents with scores."""
    return await run_in_threadpool(handle_retrieve, req)


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service status and the list of in-memory index IDs."""
    return handle_health()
