from dataclasses import dataclass
from typing import Protocol

from src.types import Document


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    provider: str = ""
    model: str = ""


class GeneratorLike(Protocol):
    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        ...
