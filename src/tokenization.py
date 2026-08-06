"""Token encoder selection with an offline-safe fallback.

``tiktoken`` downloads some encoding tables on first use. That is desirable
when the resource is available, but ingestion and module imports must not fail
only because the machine is temporarily offline.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import tiktoken

logger = logging.getLogger(__name__)


class TextEncoder(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class UnicodeCodepointEncoder:
    """Reversible fallback used only when a tiktoken table cannot be loaded."""

    name = "unicode-codepoint-fallback"

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@lru_cache(maxsize=8)
def get_tokenizer(encoding_name: str = "cl100k_base") -> TextEncoder:
    """Return a cached encoder without making offline operation fail."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except ValueError:
        # Unknown encoding names are configuration errors, not offline failures.
        raise
    except Exception as exc:  # tiktoken may raise requests/urllib errors here
        logger.warning(
            "Could not load tiktoken encoding %s; using offline fallback: %s",
            encoding_name,
            exc,
        )
        return UnicodeCodepointEncoder()
