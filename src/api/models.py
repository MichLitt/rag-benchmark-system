from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    index_id: str = "default"


class RetrievedPassage(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    source: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    index_id: str
    passages: list[RetrievedPassage]


class HealthResponse(BaseModel):
    status: str
    loaded_indexes: list[str]
