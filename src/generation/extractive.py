from src.generation.base import GenerationResult
from src.types import Document


class ExtractiveGenerator:
    """
    Bootstrap generator for local testing.
    Picks the first retrieved chunk as the answer proxy.
    """

    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        if not contexts:
            return GenerationResult(
                text="I do not know.",
                provider="local",
                model="extractive",
            )
        return GenerationResult(
            text=contexts[0].text,
            provider="local",
            model="extractive",
        )
