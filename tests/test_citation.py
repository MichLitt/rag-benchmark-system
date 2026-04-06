"""
A3 NLI citation evaluation tests.

All tests use a lightweight mock scorer — the actual HHEM model is never
downloaded here.  Covers:

  - _page_range_intersects: exact, tolerance, cross-page, None, empty-gold
  - _split_sentences: basic, empty, single sentence
  - CitationEvaluator:
      answer_attribution_rate (all/none/partial)
      supporting_passage_hit (all/none/partial)
      page_grounding_accuracy (correct/wrong/tolerance/no-gold/no-supporting)
      empty answer, empty passages, custom threshold, count consistency
  - score_faithfulness_nli (integration with faithfulness.py)
  - RunExampleResult citation fields (new defaults, round-trip)
  - HHEMScorer: protocol compliance (structural type check without loading model)
"""
from __future__ import annotations

import pytest

from src.evaluation.citation import (
    CitationEvaluator,
    CitationResult,
    NLI_THRESHOLD,
    _page_range_intersects,
    _split_sentences,
)
from src.types import Document, RunExampleResult


# ---------------------------------------------------------------------------
# Controllable mock scorer
# ---------------------------------------------------------------------------

class _FixedScorer:
    """Returns a fixed score or looks up (premise, hypothesis) in a dict."""

    def __init__(
        self,
        default: float = 0.8,
        pairs: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self._default = default
        self._pairs = pairs or {}

    def score(self, premise: str, hypothesis: str) -> float:
        return self._pairs.get((premise, hypothesis), self._default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docs(n: int, page_offset: int = 1) -> list[Document]:
    return [
        Document(
            doc_id=f"p{i}",
            text=f"passage text {i}",
            title="T",
            page_start=page_offset + i,
            page_end=page_offset + i,
        )
        for i in range(n)
    ]


def _make_run_result(**kwargs) -> RunExampleResult:
    defaults = dict(
        query_id="q1", predicted_answer="ans", gold_answers=["ans"],
        retrieved_doc_ids=[], retrieved_titles=[], unique_retrieved_titles=0,
        retrieval_latency_ms=0.0, rerank_latency_ms=0.0,
        generation_latency_ms=0.0, approx_input_tokens=0,
        approx_output_tokens=0, is_em=True, f1=1.0, recall_at_k=1.0,
        raw_candidate_count=0, dedup_candidate_count=0,
        duplicate_candidates_removed=0,
    )
    defaults.update(kwargs)
    return RunExampleResult(**defaults)


# ---------------------------------------------------------------------------
# _page_range_intersects
# ---------------------------------------------------------------------------

def test_page_intersects_exact_match():
    assert _page_range_intersects(3, 3, {3}, tolerance=0) is True


def test_page_intersects_no_overlap_zero_tolerance():
    assert _page_range_intersects(3, 3, {5}, tolerance=0) is False


def test_page_intersects_within_tolerance():
    # page 4 expanded by ±1 → [3,5] overlaps gold={5}
    assert _page_range_intersects(4, 4, {5}, tolerance=1) is True


def test_page_intersects_just_outside_tolerance():
    # page 3 expanded by ±1 → [2,4] does NOT overlap gold={5}
    assert _page_range_intersects(3, 3, {5}, tolerance=1) is False


def test_page_intersects_cross_page_with_tolerance():
    # passage spans [3,4], expanded ±1 → [2,5], gold={5} → overlap
    assert _page_range_intersects(3, 4, {5}, tolerance=1) is True


def test_page_intersects_none_page_start():
    assert _page_range_intersects(None, 3, {3}, tolerance=0) is False


def test_page_intersects_none_page_end():
    assert _page_range_intersects(3, None, {3}, tolerance=0) is False


def test_page_intersects_empty_gold():
    assert _page_range_intersects(3, 3, set(), tolerance=0) is False


def test_page_intersects_multi_gold():
    assert _page_range_intersects(5, 5, {1, 2, 5, 9}, tolerance=0) is True


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_two():
    sents = _split_sentences("Paris is the capital of France. London is in England.")
    assert len(sents) == 2
    assert any("Paris" in s for s in sents)
    assert any("London" in s for s in sents)


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


def test_split_sentences_whitespace_only():
    assert _split_sentences("   ") == []


def test_split_sentences_single():
    sents = _split_sentences("Just one sentence here.")
    assert len(sents) == 1


def test_split_sentences_strips_blanks():
    sents = _split_sentences("First.  Second.")
    assert all(s.strip() for s in sents)


# ---------------------------------------------------------------------------
# CitationEvaluator — answer_attribution_rate
# ---------------------------------------------------------------------------

def test_attribution_all_attributed():
    """All sentences have a supporting passage → rate = 1.0."""
    ev = CitationEvaluator(_FixedScorer(default=0.9))
    result = ev.evaluate("First sentence. Second sentence.", _docs(3))
    assert result.answer_attribution_rate == pytest.approx(1.0)
    assert result.attributed_sentence_count == result.total_sentence_count


def test_attribution_none_attributed():
    """No sentence has a supporting passage → rate = 0.0."""
    ev = CitationEvaluator(_FixedScorer(default=0.1))
    result = ev.evaluate("First sentence. Second sentence.", _docs(3))
    assert result.answer_attribution_rate == pytest.approx(0.0)
    assert result.attributed_sentence_count == 0


def test_attribution_partial():
    """One of two sentences supported → rate = 0.5."""
    sent_a = "Supported sentence."
    sent_b = "Unsupported sentence."
    passage_text = "supported text"

    # Only (passage, sent_a) exceeds threshold
    pairs = {
        (passage_text, sent_a): 0.9,
        (passage_text, sent_b): 0.1,
    }
    ev = CitationEvaluator(_FixedScorer(default=0.1, pairs=pairs))
    result = ev.evaluate(f"{sent_a} {sent_b}", [Document(doc_id="p0", text=passage_text, title="")])
    assert result.answer_attribution_rate == pytest.approx(0.5)
    assert result.attributed_sentence_count == 1
    assert result.total_sentence_count == 2


# ---------------------------------------------------------------------------
# CitationEvaluator — supporting_passage_hit
# ---------------------------------------------------------------------------

def test_passage_hit_all_supporting():
    ev = CitationEvaluator(_FixedScorer(default=0.9))
    result = ev.evaluate("An answer.", _docs(4))
    assert result.supporting_passage_hit == pytest.approx(1.0)
    assert result.supporting_passage_count == 4


def test_passage_hit_none_supporting():
    ev = CitationEvaluator(_FixedScorer(default=0.1))
    result = ev.evaluate("An answer.", _docs(4))
    assert result.supporting_passage_hit == pytest.approx(0.0)
    assert result.supporting_passage_count == 0


def test_passage_hit_half_supporting():
    """Passages 0 and 2 support; 1 and 3 do not → hit = 0.5."""
    docs = [Document(doc_id=f"d{i}", text=f"text{i}", title="T") for i in range(4)]
    sentence = "the answer sentence."
    pairs = {
        ("text0", sentence): 0.9,
        ("text1", sentence): 0.1,
        ("text2", sentence): 0.8,
        ("text3", sentence): 0.2,
    }
    ev = CitationEvaluator(_FixedScorer(default=0.1, pairs=pairs))
    result = ev.evaluate(sentence, docs)
    assert result.supporting_passage_hit == pytest.approx(0.5)
    assert result.supporting_passage_count == 2
    assert result.total_passage_count == 4


# ---------------------------------------------------------------------------
# CitationEvaluator — page_grounding_accuracy
# ---------------------------------------------------------------------------

def test_page_grounding_correct_page():
    doc = Document(doc_id="p0", text="text", title="T", page_start=3, page_end=3)
    ev = CitationEvaluator(_FixedScorer(default=0.9))
    result = ev.evaluate("answer.", [doc], gold_page_set={3})
    assert result.page_grounding_accuracy == pytest.approx(1.0)
    assert result.grounded_supporting_count == 1


def test_page_grounding_tolerance_adjacent():
    """passage on page 4, gold on page 5, tolerance=1 → should be grounded."""
    doc = Document(doc_id="p0", text="text", title="T", page_start=4, page_end=4)
    ev = CitationEvaluator(_FixedScorer(default=0.9), page_tolerance=1)
    result = ev.evaluate("answer.", [doc], gold_page_set={5})
    assert result.page_grounding_accuracy == pytest.approx(1.0)


def test_page_grounding_wrong_page():
    """passage on page 10, gold on page 1, tolerance=1 → not grounded."""
    doc = Document(doc_id="p0", text="text", title="T", page_start=10, page_end=10)
    ev = CitationEvaluator(_FixedScorer(default=0.9), page_tolerance=1)
    result = ev.evaluate("answer.", [doc], gold_page_set={1})
    assert result.page_grounding_accuracy == pytest.approx(0.0)
    assert result.grounded_supporting_count == 0


def test_page_grounding_partial():
    """2 supporting passages: one on correct page, one wrong → accuracy = 0.5."""
    docs = [
        Document(doc_id="correct", text="correct", title="T", page_start=5, page_end=5),
        Document(doc_id="wrong",   text="wrong",   title="T", page_start=99, page_end=99),
    ]
    ev = CitationEvaluator(_FixedScorer(default=0.9), page_tolerance=0)
    result = ev.evaluate("answer.", docs, gold_page_set={5})
    assert result.page_grounding_accuracy == pytest.approx(0.5)
    assert result.grounded_supporting_count == 1
    assert result.supporting_passage_count == 2


def test_page_grounding_no_gold_returns_none():
    doc = Document(doc_id="p0", text="text", title="T", page_start=3, page_end=3)
    ev = CitationEvaluator(_FixedScorer(default=0.9))
    result = ev.evaluate("answer.", [doc], gold_page_set=None)
    assert result.page_grounding_accuracy is None


def test_page_grounding_no_supporting_returns_none():
    """No supporting passage → page_grounding_accuracy is None (not 0/0)."""
    doc = Document(doc_id="p0", text="text", title="T", page_start=3, page_end=3)
    ev = CitationEvaluator(_FixedScorer(default=0.1))  # below threshold
    result = ev.evaluate("answer.", [doc], gold_page_set={3})
    assert result.page_grounding_accuracy is None


def test_page_grounding_no_page_metadata():
    """Supporting passage with page_start=None → not grounded."""
    doc = Document(doc_id="p0", text="text", title="T")  # no page fields
    ev = CitationEvaluator(_FixedScorer(default=0.9), page_tolerance=1)
    result = ev.evaluate("answer.", [doc], gold_page_set={3})
    assert result.page_grounding_accuracy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CitationEvaluator — edge cases
# ---------------------------------------------------------------------------

def test_empty_answer_returns_none_metrics():
    ev = CitationEvaluator(_FixedScorer())
    result = ev.evaluate("", _docs(3))
    assert result.answer_attribution_rate is None
    assert result.supporting_passage_hit is None
    assert result.page_grounding_accuracy is None


def test_whitespace_answer_returns_none():
    ev = CitationEvaluator(_FixedScorer())
    result = ev.evaluate("   \n  ", _docs(2))
    assert result.answer_attribution_rate is None


def test_empty_passages_returns_none_hit():
    """0 passages → supporting_passage_hit is None (0/0 is undefined, not 0.0)."""
    ev = CitationEvaluator(_FixedScorer())
    result = ev.evaluate("Some answer.", [])
    assert result.supporting_passage_hit is None
    assert result.total_passage_count == 0


def test_custom_threshold_boundary():
    """Score of 0.65: above 0.5 threshold → attributed; below 0.7 → not."""
    ev_low = CitationEvaluator(_FixedScorer(default=0.65), threshold=0.5)
    ev_high = CitationEvaluator(_FixedScorer(default=0.65), threshold=0.7)
    assert ev_low.evaluate("answer.", _docs(1)).answer_attribution_rate == pytest.approx(1.0)
    assert ev_high.evaluate("answer.", _docs(1)).answer_attribution_rate == pytest.approx(0.0)


def test_count_consistency():
    """Fraction fields must be consistent with their diagnostic counts."""
    ev = CitationEvaluator(_FixedScorer(default=0.9))
    result = ev.evaluate("Sentence one. Sentence two.", _docs(3))
    assert result.total_passage_count == 3
    assert result.total_sentence_count == 2
    assert result.supporting_passage_count / result.total_passage_count == pytest.approx(
        result.supporting_passage_hit
    )
    assert result.attributed_sentence_count / result.total_sentence_count == pytest.approx(
        result.answer_attribution_rate
    )


def test_single_passage_single_sentence():
    """Minimal 1×1 case: one passage, one sentence."""
    doc = Document(doc_id="d0", text="Paris is in France.", title="T",
                   page_start=1, page_end=1)
    ev = CitationEvaluator(_FixedScorer(default=0.9))
    result = ev.evaluate("Paris is in France.", [doc], gold_page_set={1})
    assert result.answer_attribution_rate == pytest.approx(1.0)
    assert result.supporting_passage_hit == pytest.approx(1.0)
    assert result.page_grounding_accuracy == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_faithfulness_nli (faithfulness.py integration)
# ---------------------------------------------------------------------------

def test_score_faithfulness_nli_above_threshold():
    from src.evaluation.faithfulness import score_faithfulness_nli
    score = score_faithfulness_nli(
        answer="Paris is the capital.",
        context_texts=["Paris is the capital of France."],
        scorer=_FixedScorer(default=0.9),
    )
    assert score == pytest.approx(1.0)


def test_score_faithfulness_nli_below_threshold():
    from src.evaluation.faithfulness import score_faithfulness_nli
    score = score_faithfulness_nli(
        answer="London is the capital.",
        context_texts=["Berlin is the capital of Germany."],
        scorer=_FixedScorer(default=0.1),
    )
    assert score == pytest.approx(0.0)


def test_score_faithfulness_nli_empty_answer():
    from src.evaluation.faithfulness import score_faithfulness_nli
    assert score_faithfulness_nli("", ["some context"], scorer=_FixedScorer()) == 0.0


def test_score_faithfulness_nli_empty_contexts():
    from src.evaluation.faithfulness import score_faithfulness_nli
    assert score_faithfulness_nli("some answer.", [], scorer=_FixedScorer()) == 0.0


# ---------------------------------------------------------------------------
# RunExampleResult — citation field defaults + assignment
# ---------------------------------------------------------------------------

def test_run_example_result_citation_defaults_none():
    r = _make_run_result()
    assert r.answer_attribution_rate is None
    assert r.supporting_passage_hit is None
    assert r.page_grounding_accuracy is None


def test_run_example_result_citation_field_assignment():
    r = _make_run_result(
        answer_attribution_rate=0.75,
        supporting_passage_hit=0.6,
        page_grounding_accuracy=1.0,
    )
    assert r.answer_attribution_rate == pytest.approx(0.75)
    assert r.supporting_passage_hit == pytest.approx(0.6)
    assert r.page_grounding_accuracy == pytest.approx(1.0)


def test_run_example_result_citation_accepts_zero():
    r = _make_run_result(answer_attribution_rate=0.0)
    assert r.answer_attribution_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# NLIScorer protocol — structural check (no model download)
# ---------------------------------------------------------------------------

def test_fixed_scorer_satisfies_nli_protocol():
    from src.evaluation.hhem_scorer import NLIScorer
    scorer = _FixedScorer()
    assert isinstance(scorer, NLIScorer), (
        "_FixedScorer must satisfy the NLIScorer protocol "
        "(has score(premise, hypothesis) -> float)"
    )


def test_nli_scorer_protocol_rejects_bad_obj():
    from src.evaluation.hhem_scorer import NLIScorer

    class _Bad:
        pass

    assert not isinstance(_Bad(), NLIScorer)
