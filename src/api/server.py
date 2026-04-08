from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from src.api.handlers import handle_retrieve
from src.api.index_registry import IndexRegistry
from src.api.models import HealthResponse, RetrieveRequest, RetrieveResponse

_registry = IndexRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load indexes from INDEX_CONFIG_<ID> environment variables at startup.

    Example:
        INDEX_CONFIG_DEFAULT=config/wiki18_21m_sharded.yaml
        INDEX_CONFIG_MYPDF=data/indexes/my_pdf_index/config.yaml
    """
    prefix = "INDEX_CONFIG_"
    for key, val in os.environ.items():
        if key.startswith(prefix):
            index_id = key[len(prefix):].lower()
            try:
                _registry.load_from_config(index_id, val)
                print(f"[startup] Loaded index '{index_id}' from {val}")
            except Exception as exc:
                print(f"[startup] WARNING: Failed to load index '{index_id}' from {val}: {exc}")
    yield
    _registry.close_all()


app = FastAPI(
    title="Agent Knowledge Retrieval API",
    description="HTTP retrieval service for the RAG Benchmark knowledge subsystem.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    try:
        return handle_retrieve(req, _registry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", loaded_indexes=_registry.list_ids())
