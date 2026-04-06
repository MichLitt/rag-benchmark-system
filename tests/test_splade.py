"""Tests for SPLADE sparse retriever (B1).

All tests use a FakeSparseEncoder that assigns deterministic sparse vectors
based on word-overlap, so no HuggingFace model download is needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from src.retrieval.hybrid import DenseSpladeHybridRetriever, rrf_fusion
from src.retrieval.splade import (
    SPLADERetriever,
    SparseEncoder,
    build_splade_index,
)
from src.types import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100  # tiny fake vocabulary


def _make_doc(doc_id: str, text: str, title: str = "") -> Document:
    return Document(doc_id=doc_id, text=text, title=title)


class FakeSparseEncoder:
    """Deterministic sparse encoder: activates vocab dims matching word hashes."""

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self._vocab_size, dtype=np.float32)
        for word in text.lower().split():
            idx = hash(word) % self._vocab_size
            vec[idx] = max(vec[idx], 1.0)
        return vec

    def encode_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> scipy.sparse.csr_matrix:
        rows = [scipy.sparse.csr_matrix(self.encode(t)) for t in texts]
        if not rows:
            return scipy.sparse.csr_matrix((0, self._vocab_size), dtype=np.float32)
        return scipy.sparse.vstack(rows)


def _make_retriever(docs: list[Document]) -> SPLADERetriever:
    enc = FakeSparseEncoder()
    texts = [f"{d.title}\n{d.text}".strip() for d in docs]
    matrix = enc.encode_batch(texts)
    return SPLADERetriever(matrix, docs, enc)


# ---------------------------------------------------------------------------
# SparseEncoder protocol
# ---------------------------------------------------------------------------

def test_fake_encoder_satisfies_sparse_encoder_protocol():
    enc = FakeSparseEncoder()
    assert isinstance(enc, SparseEncoder)


def test_fake_encoder_vocab_size():
    enc = FakeSparseEncoder(vocab_size=64)
    assert enc.vocab_size == 64


def test_fake_encoder_encode_shape():
    enc = FakeSparseEncoder()
    vec = enc.encode("hello world")
    assert vec.shape == (VOCAB_SIZE,)
    assert vec.dtype == np.float32


def test_fake_encoder_encode_sparse():
    enc = FakeSparseEncoder()
    vec = enc.encode("hello world")
    assert vec.sum() > 0
    assert (vec >= 0).all()


def test_fake_encoder_encode_batch_shape():
    enc = FakeSparseEncoder()
    texts = ["hello world", "foo bar baz"]
    mat = enc.encode_batch(texts)
    assert mat.shape == (2, VOCAB_SIZE)


def test_fake_encoder_encode_batch_empty():
    enc = FakeSparseEncoder()
    mat = enc.encode_batch([])
    assert mat.shape == (0, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# build_splade_index
# ---------------------------------------------------------------------------

def test_build_index_shape():
    docs = [_make_doc("a", "hello world"), _make_doc("b", "foo bar")]
    enc = FakeSparseEncoder()
    matrix = build_splade_index(docs, enc)
    assert matrix.shape == (2, VOCAB_SIZE)


def test_build_index_empty_docs():
    enc = FakeSparseEncoder()
    matrix = build_splade_index([], enc)
    assert matrix.shape == (0, VOCAB_SIZE)


def test_build_index_uses_title_and_text():
    docs = [_make_doc("a", text="unique_word_x", title="unique_word_y")]
    enc = FakeSparseEncoder()
    matrix = build_splade_index(docs, enc)
    # Both words should activate dimensions
    dense = np.asarray(matrix.todense()).flatten()
    assert dense.sum() > 0


# ---------------------------------------------------------------------------
# SPLADERetriever
# ---------------------------------------------------------------------------

def test_retrieve_returns_top_k():
    docs = [_make_doc(f"d{i}", f"document about topic_{i}") for i in range(5)]
    ret = _make_retriever(docs)
    results = ret.retrieve("topic_0", top_k=3)
    assert len(results) == 3


def test_retrieve_top_k_zero():
    docs = [_make_doc("a", "hello")]
    ret = _make_retriever(docs)
    assert ret.retrieve("hello", top_k=0) == []


def test_retrieve_empty_docs():
    enc = FakeSparseEncoder()
    ret = SPLADERetriever(enc.encode_batch([]), [], enc)
    assert ret.retrieve("hello", top_k=5) == []


def test_retrieve_respects_top_k_cap():
    docs = [_make_doc(f"d{i}", "same text for all") for i in range(3)]
    ret = _make_retriever(docs)
    results = ret.retrieve("same text", top_k=10)
    assert len(results) == 3  # capped at corpus size


def test_retrieve_relevant_doc_ranks_first():
    """Doc containing exact query words should score highest."""
    docs = [
        _make_doc("irrelevant", "completely unrelated words xyz"),
        _make_doc("relevant", "paris france capital city"),
    ]
    ret = _make_retriever(docs)
    results = ret.retrieve("paris france capital", top_k=2)
    assert results[0].doc_id == "relevant"


def test_retrieve_with_scores_length_matches():
    docs = [_make_doc(f"d{i}", f"text {i}") for i in range(4)]
    ret = _make_retriever(docs)
    result_docs, scores = ret.retrieve_with_scores("text", top_k=3)
    assert len(result_docs) == len(scores) == 3


def test_retrieve_with_scores_positive():
    docs = [_make_doc("a", "hello world"), _make_doc("b", "other content")]
    ret = _make_retriever(docs)
    _, scores = ret.retrieve_with_scores("hello world", top_k=2)
    assert all(s >= 0 for s in scores)


def test_retrieve_with_scores_monotone_decreasing():
    docs = [_make_doc(f"d{i}", f"cat dog bird fish item {i}") for i in range(5)]
    ret = _make_retriever(docs)
    _, scores = ret.retrieve_with_scores("cat dog bird", top_k=5)
    for a, b in zip(scores, scores[1:]):
        assert a >= b, f"scores not monotone: {scores}"


# ---------------------------------------------------------------------------
# rrf_fusion (standalone function)
# ---------------------------------------------------------------------------

def test_rrf_fusion_combines_unique_docs():
    a = [_make_doc("x", "text"), _make_doc("y", "text")]
    b = [_make_doc("z", "text"), _make_doc("y", "text")]
    result = rrf_fusion(a, b, top_k=10)
    ids = {d.doc_id for d in result}
    assert ids == {"x", "y", "z"}


def test_rrf_fusion_shared_doc_gets_higher_score():
    """A doc appearing in both lists should rank above docs in only one list."""
    shared = _make_doc("shared", "text")
    a = [shared, _make_doc("a_only", "text")]
    b = [shared, _make_doc("b_only", "text")]
    result = rrf_fusion(a, b, top_k=3)
    assert result[0].doc_id == "shared"


def test_rrf_fusion_top_k_respected():
    docs_a = [_make_doc(f"a{i}", "x") for i in range(10)]
    docs_b = [_make_doc(f"b{i}", "x") for i in range(10)]
    result = rrf_fusion(docs_a, docs_b, top_k=5)
    assert len(result) == 5


def test_rrf_fusion_empty_lists():
    result = rrf_fusion([], [], top_k=5)
    assert result == []


def test_rrf_fusion_one_empty_list():
    docs = [_make_doc("a", "x"), _make_doc("b", "x")]
    result = rrf_fusion(docs, [], top_k=5)
    assert {d.doc_id for d in result} == {"a", "b"}


def test_rrf_fusion_alpha_weight():
    """With alpha=1.0 (weight_a=1, weight_b=0), only list_a scores matter."""
    a = [_make_doc("top_a", "x"), _make_doc("mid_a", "x")]
    b = [_make_doc("top_b", "x"), _make_doc("top_a", "x")]
    result = rrf_fusion(a, b, weight_a=1.0, weight_b=0.0, top_k=2)
    assert result[0].doc_id == "top_a"


# ---------------------------------------------------------------------------
# DenseSpladeHybridRetriever
# ---------------------------------------------------------------------------

class _FakeDenseRetriever:
    """Minimal dense retriever stub for hybrid tests."""
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return self._docs[:top_k]


def test_dense_splade_hybrid_returns_top_k():
    docs_dense = [_make_doc(f"d{i}", f"dense doc {i}") for i in range(5)]
    docs_splade = [_make_doc(f"s{i}", f"splade doc {i}") for i in range(5)]
    enc = FakeSparseEncoder()
    matrix = enc.encode_batch([f"splade doc {i}" for i in range(5)])
    splade = SPLADERetriever(matrix, docs_splade, enc)
    dense = _FakeDenseRetriever(docs_dense)
    hybrid = DenseSpladeHybridRetriever(dense, splade, alpha=0.5)
    results = hybrid.retrieve("splade dense doc", top_k=4)
    assert len(results) == 4


def test_dense_splade_hybrid_alpha_validation():
    enc = FakeSparseEncoder()
    splade = SPLADERetriever(enc.encode_batch([]), [], enc)
    dense = _FakeDenseRetriever([])
    with pytest.raises(ValueError, match="alpha"):
        DenseSpladeHybridRetriever(dense, splade, alpha=1.5)


def test_dense_splade_hybrid_rrf_k_validation():
    enc = FakeSparseEncoder()
    splade = SPLADERetriever(enc.encode_batch([]), [], enc)
    dense = _FakeDenseRetriever([])
    with pytest.raises(ValueError, match="rrf_k"):
        DenseSpladeHybridRetriever(dense, splade, rrf_k=0)


def test_dense_splade_hybrid_zero_top_k():
    enc = FakeSparseEncoder()
    splade = SPLADERetriever(enc.encode_batch([]), [], enc)
    dense = _FakeDenseRetriever([])
    hybrid = DenseSpladeHybridRetriever(dense, splade)
    assert hybrid.retrieve("q", top_k=0) == []


def test_dense_splade_hybrid_merges_unique_ids():
    docs_dense = [_make_doc("d1", "topic alpha")]
    docs_splade = [_make_doc("s1", "topic beta")]
    enc = FakeSparseEncoder()
    matrix = enc.encode_batch(["topic beta"])
    splade = SPLADERetriever(matrix, docs_splade, enc)
    dense = _FakeDenseRetriever(docs_dense)
    hybrid = DenseSpladeHybridRetriever(dense, splade, candidate_k=10)
    results = hybrid.retrieve("topic", top_k=5)
    ids = {d.doc_id for d in results}
    assert "d1" in ids
    assert "s1" in ids
