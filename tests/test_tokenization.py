from __future__ import annotations

import pytest

from src import tokenization


def test_offline_fallback_is_reversible(monkeypatch):
    tokenization.get_tokenizer.cache_clear()
    monkeypatch.setattr(
        tokenization.tiktoken,
        "get_encoding",
        lambda _name: (_ for _ in ()).throw(OSError("offline")),
    )

    encoder = tokenization.get_tokenizer("offline-test")
    text = "Agent 检索 works"

    assert isinstance(encoder, tokenization.UnicodeCodepointEncoder)
    assert encoder.decode(encoder.encode(text)) == text
    tokenization.get_tokenizer.cache_clear()


def test_unknown_encoding_is_not_silently_downgraded(monkeypatch):
    tokenization.get_tokenizer.cache_clear()
    monkeypatch.setattr(
        tokenization.tiktoken,
        "get_encoding",
        lambda _name: (_ for _ in ()).throw(ValueError("unknown encoding")),
    )

    with pytest.raises(ValueError, match="unknown encoding"):
        tokenization.get_tokenizer("typo")

    tokenization.get_tokenizer.cache_clear()
