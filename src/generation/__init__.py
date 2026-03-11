"""Generation modules."""

from src.generation.base import GenerationResult, GeneratorLike
from src.generation.extractive import ExtractiveGenerator
from src.generation.factory import build_generator
from src.generation.openai_compatible import OpenAICompatibleGenerator

__all__ = [
    "ExtractiveGenerator",
    "GenerationResult",
    "GeneratorLike",
    "OpenAICompatibleGenerator",
    "build_generator",
]
