from __future__ import annotations

from typing import Protocol


class QueryExpanderLike(Protocol):
    def expand(self, question: str) -> str:
        ...

    def expand_queries(self, question: str) -> list[str]:
        ...
