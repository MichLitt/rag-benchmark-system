"""Tests for setwise LLM reranker and two-stage reranking pipeline (B2).

All tests use FakeTextGenerator and FakeCrossEncoder so no model download
is needed.
"""
from __future__ import annotations

import pytest

from src.reranking.pipeline import TwoStageReranker
from src.reranking.setwise import SetwiseLLMReranker, TextGenerator
from src.types import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(doc_id: str, text: str = "text") -> Document:
    return Document(doc_id=doc_id, text=text, title="")


class FixedRankGenerator:
    """Returns a preset comma-separated ranking regardless of prompt."""

    def __init__(self, ranking: str) -> None:
        self._ranking = ranking

    def generate(self, prompt: str) -> str:
        return self._ranking


class IdentityRankGenerator:
    """Returns indices in original order (1,2,3,...)."""

    def generate(self, prompt: str) -> str:
        # Extract passage count from prompt "[N]" markers
        import re
        markers = re.findall(r"\[(\d+)\]", prompt)
        return ",".join(markers)


class ReverseRankGenerator:
    """Returns indices in reverse order (N,...,2,1)."""

    def generate(self, prompt: str) -> str:
        import re
        markers = re.findall(r"\[(\d+)\]", prompt)
        return ",".join(reversed(markers))


class FailingGenerator:
    """Always raises an exception."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("model unavailable")


class FakeCrossEncoder:
    """Stub cross-encoder that just returns the first top_k docs unchanged."""

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        return docs[:top_k]


# ---------------------------------------------------------------------------
# TextGenerator protocol
# ---------------------------------------------------------------------------

def test_fixed_generator_satisfies_protocol():
    gen = FixedRankGenerator("1,2,3")
    assert isinstance(gen, TextGenerator)


def test_identity_generator_satisfies_protocol():
    assert isinstance(IdentityRankGenerator(), TextGenerator)


# ---------------------------------------------------------------------------
# SetwiseLLMReranker — basic construction
# ---------------------------------------------------------------------------

def test_setwise_rejects_zero_max_candidates():
    with pytest.raises(ValueError, match="max_candidates"):
        SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=0)


def test_setwise_rejects_negative_max_candidates():
    with pytest.raises(ValueError, match="max_candidates"):
        SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=-1)


# ---------------------------------------------------------------------------
# _parse_ranking (tested via rerank behaviour)
# ---------------------------------------------------------------------------

def test_parse_ranking_valid_reversal():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    sw = SetwiseLLMReranker(FixedRankGenerator("3,2,1"))
    result = sw.rerank("q", docs, top_k=3)
    assert [d.doc_id for d in result] == ["c", "b", "a"]


def test_parse_ranking_partial_response_appends_missing():
    """LLM only lists index 2; indices 1 and 3 should be appended."""
    docs = [_doc("a"), _doc("b"), _doc("c")]
    sw = SetwiseLLMReranker(FixedRankGenerator("2"))
    result = sw.rerank("q", docs, top_k=3)
    assert result[0].doc_id == "b"
    # remaining in original order
    assert result[1].doc_id == "a"
    assert result[2].doc_id == "c"


def test_parse_ranking_out_of_range_ignored():
    """Index 99 is out of range for 3 docs; should be ignored."""
    docs = [_doc("a"), _doc("b"), _doc("c")]
    sw = SetwiseLLMReranker(FixedRankGenerator("99,2,1,3"))
    result = sw.rerank("q", docs, top_k=3)
    assert {d.doc_id for d in result} == {"a", "b", "c"}


def test_parse_ranking_duplicate_indices_deduplicated():
    """Duplicate index 1 in response should only appear once."""
    docs = [_doc("a"), _doc("b")]
    sw = SetwiseLLMReranker(FixedRankGenerator("1,1,2"))
    result = sw.rerank("q", docs, top_k=2)
    assert len(result) == 2
    assert result[0].doc_id == "a"
    assert result[1].doc_id == "b"


def test_parse_ranking_empty_response_falls_back_to_original_order():
    docs = [_doc("x"), _doc("y"), _doc("z")]
    sw = SetwiseLLMReranker(FixedRankGenerator(""))
    result = sw.rerank("q", docs, top_k=3)
    assert [d.doc_id for d in result] == ["x", "y", "z"]


def test_generator_exception_falls_back_to_original_order():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    sw = SetwiseLLMReranker(FailingGenerator())
    result = sw.rerank("q", docs, top_k=3)
    assert [d.doc_id for d in result] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# rerank / rerank_with_scores
# ---------------------------------------------------------------------------

def test_rerank_returns_top_k():
    docs = [_doc(f"d{i}") for i in range(8)]
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    result = sw.rerank("query", docs, top_k=4)
    assert len(result) == 4


def test_rerank_zero_top_k():
    docs = [_doc("a"), _doc("b")]
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    assert sw.rerank("q", docs, top_k=0) == []


def test_rerank_empty_docs():
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    assert sw.rerank("q", [], top_k=5) == []


def test_rerank_top_k_larger_than_corpus():
    docs = [_doc("a"), _doc("b")]
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    result = sw.rerank("q", docs, top_k=10)
    assert len(result) == 2


def test_rerank_with_scores_scores_decrease():
    docs = [_doc(f"d{i}") for i in range(5)]
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    _, scores = sw.rerank_with_scores("q", docs, top_k=5)
    assert scores == sorted(scores, reverse=True)


def test_rerank_with_scores_first_score_is_one():
    docs = [_doc("a")]
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    _, scores = sw.rerank_with_scores("q", docs, top_k=1)
    assert scores[0] == pytest.approx(1.0)


def test_rerank_with_scores_formula():
    """Score for rank r should be 1/(r+1)."""
    docs = [_doc(f"d{i}") for i in range(3)]
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    _, scores = sw.rerank_with_scores("q", docs, top_k=3)
    assert scores[0] == pytest.approx(1.0 / 1)
    assert scores[1] == pytest.approx(1.0 / 2)
    assert scores[2] == pytest.approx(1.0 / 3)


def test_rerank_respects_llm_ordering():
    """ReverseRankGenerator should produce docs in reverse order."""
    docs = [_doc("first"), _doc("second"), _doc("third")]
    sw = SetwiseLLMReranker(ReverseRankGenerator(), max_candidates=10)
    result = sw.rerank("q", docs, top_k=3)
    assert [d.doc_id for d in result] == ["third", "second", "first"]


# ---------------------------------------------------------------------------
# Sliding-window (>max_candidates docs)
# ---------------------------------------------------------------------------

def test_sliding_window_invoked_for_large_input():
    """With 5 docs and max_candidates=3, sliding window must be used."""
    docs = [_doc(f"d{i}") for i in range(5)]
    sw = SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=3)
    result = sw.rerank("q", docs, top_k=5)
    assert len(result) == 5
    assert {d.doc_id for d in result} == {f"d{i}" for i in range(5)}


def test_sliding_window_no_duplicates():
    docs = [_doc(f"d{i}") for i in range(6)]
    sw = SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=4)
    result = sw.rerank("q", docs, top_k=6)
    assert len(result) == len(set(d.doc_id for d in result))


def test_sliding_window_top_k_respected():
    docs = [_doc(f"d{i}") for i in range(8)]
    sw = SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=3)
    result = sw.rerank("q", docs, top_k=4)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# TwoStageReranker
# ---------------------------------------------------------------------------

def test_two_stage_rejects_invalid_coarse_top_k():
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    with pytest.raises(ValueError, match="coarse_top_k"):
        TwoStageReranker(ce, sw, coarse_top_k=0)


def test_two_stage_rerank_returns_top_k():
    docs = [_doc(f"d{i}") for i in range(10)]
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=10)
    ts = TwoStageReranker(ce, sw, coarse_top_k=5)
    result = ts.rerank("query", docs, top_k=3)
    assert len(result) == 3


def test_two_stage_rerank_empty_docs():
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    ts = TwoStageReranker(ce, sw)
    assert ts.rerank("q", [], top_k=5) == []


def test_two_stage_rerank_zero_top_k():
    docs = [_doc("a"), _doc("b")]
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(IdentityRankGenerator())
    ts = TwoStageReranker(ce, sw)
    assert ts.rerank("q", docs, top_k=0) == []


def test_two_stage_with_scores_scores_decrease():
    docs = [_doc(f"d{i}") for i in range(6)]
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(IdentityRankGenerator(), max_candidates=10)
    ts = TwoStageReranker(ce, sw, coarse_top_k=6)
    _, scores = ts.rerank_with_scores("query", docs, top_k=4)
    assert scores == sorted(scores, reverse=True)


def test_two_stage_coarse_limits_stage2_input():
    """TwoStageReranker must pass exactly coarse_top_k docs to stage 2."""
    received: list[int] = []

    class CountingGenerator:
        def generate(self, prompt: str) -> str:
            import re
            markers = re.findall(r"\[(\d+)\]", prompt)
            received.append(len(markers))
            return ",".join(markers)

    docs = [_doc(f"d{i}") for i in range(20)]
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(CountingGenerator(), max_candidates=10)
    ts = TwoStageReranker(ce, sw, coarse_top_k=5)
    ts.rerank("query", docs, top_k=3)
    # Stage 2 should receive exactly 5 docs (coarse_top_k ≤ max_candidates)
    assert received[0] == 5


def test_two_stage_stage2_respects_max_candidates():
    """With coarse_top_k > max_candidates, sliding window is used."""
    received: list[int] = []

    class CountingGenerator:
        def generate(self, prompt: str) -> str:
            import re
            markers = re.findall(r"\[(\d+)\]", prompt)
            received.append(len(markers))
            return ",".join(markers)

    docs = [_doc(f"d{i}") for i in range(20)]
    ce = FakeCrossEncoder()
    sw = SetwiseLLMReranker(CountingGenerator(), max_candidates=5)
    ts = TwoStageReranker(ce, sw, coarse_top_k=12)
    ts.rerank("query", docs, top_k=3)
    # Each window call should have ≤ 5 docs
    assert all(n <= 5 for n in received)
    # Sliding window should have been called more than once
    assert len(received) > 1
