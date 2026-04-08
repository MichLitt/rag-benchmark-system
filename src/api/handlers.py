from __future__ import annotations

from src.api.index_registry import IndexRegistry
from src.api.models import RetrievedPassage, RetrieveRequest, RetrieveResponse
from src.types import Document


def _wrap_scores(docs: list[Document]) -> list[RetrievedPassage]:
    """Convert a ranked List[Document] to List[RetrievedPassage] with surrogate scores.

    The existing retriever protocol (retrieve() -> list[Document]) does not
    expose raw similarity scores. We assign a monotone rank-normalised surrogate:
    score = (n - rank + 1) / n, so rank-1 gets score 1.0 and last gets 1/n.
    This preserves ordering and gives callers a numeric score to filter on.
    """
    n = len(docs)
    passages: list[RetrievedPassage] = []
    for rank, doc in enumerate(docs, start=1):
        passages.append(RetrievedPassage(
            doc_id=doc.doc_id,
            title=doc.title,
            text=doc.text,
            score=float(n - rank + 1) / max(n, 1),
            rank=rank,
            page_start=doc.page_start,
            page_end=doc.page_end,
            section=doc.section,
            source=doc.source,
        ))
    return passages


def handle_retrieve(req: RetrieveRequest, registry: IndexRegistry) -> RetrieveResponse:
    retriever = registry.get(req.index_id)
    if retriever is None:
        raise KeyError(f"Index not found: {req.index_id!r}")
    docs: list[Document] = retriever.retrieve(req.query, req.top_k)
    return RetrieveResponse(
        query=req.query,
        index_id=req.index_id,
        passages=_wrap_scores(docs),
    )
