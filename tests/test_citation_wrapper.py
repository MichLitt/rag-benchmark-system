"""Unit tests for src/generation/citation_wrapper.py."""
from __future__ import annotations

import pytest

from src.generation.citation_wrapper import (
    CitationConstrainedGenerator,
    CitationScoringResult,
    NLI_THRESHOLD,
    parse_citation_indices,
    score_citation_precision,
)
from src.generation.base import GenerationResult
from src.types import Document


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_doc(text: str, doc_id: str = "d1", title: str = "T") -> Document:
    return Document(doc_id=doc_id, text=text, title=title)


class _FixedScorer:
    """NLI scorer that returns a fixed score for all (premise, hypothesis) pairs."""

    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, premise: str, hypothesis: str) -> float:
        return self._score


class _PerPassageScorer:
    """NLI scorer with per-passage scores keyed by passage text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, premise: str, hypothesis: str) -> float:
        return self._scores.get(premise, 0.0)


class _FixedGenerator:
    """Minimal GeneratorLike stub that returns a preset answer."""

    def __init__(self, answer: str, system_prompt: str = "default prompt") -> None:
        self.system_prompt = system_prompt
        self._answer = answer

    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        return GenerationResult(text=self._answer, provider="stub", model="stub")


# ---------------------------------------------------------------------------
# parse_citation_indices
# ---------------------------------------------------------------------------


class TestParseCitationIndices:
    def test_single_citation(self):
        assert parse_citation_indices("Answer [1].") == [1]

    def test_multiple_citations_in_brackets(self):
        assert parse_citation_indices("Fact [1, 2, 3].") == [1, 2, 3]

    def test_multiple_separate_citations(self):
        assert parse_citation_indices("A [1]. B [3].") == [1, 3]

    def test_deduplication(self):
        assert parse_citation_indices("[1] foo [1] bar [2]") == [1, 2]

    def test_sorted_output(self):
        assert parse_citation_indices("[3] [1] [2]") == [1, 2, 3]

    def test_no_citations(self):
        assert parse_citation_indices("No citations here.") == []

    def test_empty_string(self):
        assert parse_citation_indices("") == []

    def test_out_of_range_index_included(self):
        # Out-of-range filtering happens in score_citation_precision, not here.
        assert parse_citation_indices("[99]") == [99]

    def test_zero_index(self):
        # [0] is parsed — range validation is downstream.
        assert parse_citation_indices("[0]") == [0]

    def test_citation_with_spaces(self):
        assert parse_citation_indices("[1 , 2 , 3]") == [1, 2, 3]

    def test_two_digit_index(self):
        assert parse_citation_indices("[12]") == [12]


# ---------------------------------------------------------------------------
# score_citation_precision
# ---------------------------------------------------------------------------


class TestScoreCitationPrecision:
    def _contexts(self, n: int) -> list[Document]:
        return [_make_doc(f"passage text {i}", doc_id=f"d{i}") for i in range(1, n + 1)]

    def test_all_supported(self):
        contexts = self._contexts(3)
        result = score_citation_precision(
            "Answer [1] [2].", contexts, _FixedScorer(0.9)
        )
        assert result.citation_count == 2
        assert result.citation_precision == pytest.approx(1.0)
        assert result.cited_indices == [1, 2]
        assert set(result.hhem_scores.keys()) == {1, 2}

    def test_none_supported(self):
        contexts = self._contexts(3)
        result = score_citation_precision(
            "Answer [1] [2].", contexts, _FixedScorer(0.1)
        )
        assert result.citation_precision == pytest.approx(0.0)

    def test_partial_support(self):
        contexts = [
            _make_doc("supported passage"),
            _make_doc("unsupported passage"),
        ]
        scorer = _PerPassageScorer(
            {"supported passage": 0.9, "unsupported passage": 0.2}
        )
        result = score_citation_precision("A [1]. B [2].", contexts, scorer)
        assert result.citation_precision == pytest.approx(0.5)

    def test_out_of_range_citation_excluded_from_precision(self):
        contexts = self._contexts(2)
        result = score_citation_precision(
            "Answer [1] [99].", contexts, _FixedScorer(0.9)
        )
        # Only [1] is in-range → 1 supported / 1 in-range = 1.0
        assert result.citation_precision == pytest.approx(1.0)
        assert result.citation_count == 2  # both counted in total
        assert 99 not in result.hhem_scores

    def test_no_citations_returns_none_precision(self):
        contexts = self._contexts(2)
        result = score_citation_precision("No citations.", contexts, _FixedScorer(0.9))
        assert result.citation_count == 0
        assert result.citation_precision is None
        assert result.cited_indices == []

    def test_citation_with_all_out_of_range_returns_zero_precision(self):
        contexts = self._contexts(2)
        result = score_citation_precision("[99]", contexts, _FixedScorer(0.9))
        assert result.citation_precision == pytest.approx(0.0)
        assert result.citation_count == 1

    def test_threshold_boundary(self):
        contexts = self._contexts(1)
        # Exactly at threshold should be supported.
        result = score_citation_precision("[1]", contexts, _FixedScorer(NLI_THRESHOLD))
        assert result.citation_precision == pytest.approx(1.0)

    def test_just_below_threshold_not_supported(self):
        contexts = self._contexts(1)
        result = score_citation_precision(
            "[1]", contexts, _FixedScorer(NLI_THRESHOLD - 0.01)
        )
        assert result.citation_precision == pytest.approx(0.0)

    def test_empty_contexts(self):
        result = score_citation_precision("[1]", [], _FixedScorer(0.9))
        # [1] is out-of-range for empty contexts → zero precision
        assert result.citation_precision == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CitationConstrainedGenerator
# ---------------------------------------------------------------------------


class TestCitationConstrainedGenerator:
    def _gen(self, answer: str, prompt: str = "base prompt") -> _FixedGenerator:
        return _FixedGenerator(answer=answer, system_prompt=prompt)

    def test_citation_instruction_injected_into_system_prompt(self):
        from src.generation.prompts import CITATION_INSTRUCTION

        base = self._gen("Paris [1].")
        wrapper = CitationConstrainedGenerator(base_generator=base)
        assert CITATION_INSTRUCTION in base.system_prompt

    def test_citation_instruction_not_duplicated(self):
        from src.generation.prompts import CITATION_INSTRUCTION

        base = self._gen("Paris [1].", prompt="existing" + CITATION_INSTRUCTION)
        wrapper = CitationConstrainedGenerator(base_generator=base)
        assert base.system_prompt.count(CITATION_INSTRUCTION) == 1

    def test_generate_returns_base_result_unchanged(self):
        base = self._gen("Paris [1].")
        wrapper = CitationConstrainedGenerator(base_generator=base, scorer=None)
        contexts = [_make_doc("Paris is in France.")]
        result = wrapper.generate("Where is Paris?", contexts)
        assert result.text == "Paris [1]."

    def test_last_citation_result_set_after_generate(self):
        base = self._gen("Paris [1].")
        wrapper = CitationConstrainedGenerator(base_generator=base, scorer=None)
        contexts = [_make_doc("Paris is in France.")]
        assert wrapper.last_citation_result is None
        wrapper.generate("Where is Paris?", contexts)
        assert wrapper.last_citation_result is not None

    def test_scorer_disabled_when_no_scorer(self):
        base = self._gen("Paris [1].")
        wrapper = CitationConstrainedGenerator(base_generator=base, scorer=None)
        contexts = [_make_doc("Paris is in France.")]
        wrapper.generate("Where is Paris?", contexts)
        cr = wrapper.last_citation_result
        assert cr is not None
        assert cr.scorer_disabled is True
        assert cr.citation_precision is None
        assert cr.citation_count == 1

    def test_scorer_enabled_scores_precision(self):
        base = self._gen("Paris [1].")
        wrapper = CitationConstrainedGenerator(
            base_generator=base, scorer=_FixedScorer(0.9)
        )
        contexts = [_make_doc("Paris is in France.")]
        wrapper.generate("Where is Paris?", contexts)
        cr = wrapper.last_citation_result
        assert cr is not None
        assert cr.scorer_disabled is False
        assert cr.citation_precision == pytest.approx(1.0)
        assert 1 in cr.hhem_scores

    def test_no_citations_in_answer(self):
        base = self._gen("Paris.")
        wrapper = CitationConstrainedGenerator(
            base_generator=base, scorer=_FixedScorer(0.9)
        )
        contexts = [_make_doc("Paris is in France.")]
        wrapper.generate("Where is Paris?", contexts)
        cr = wrapper.last_citation_result
        assert cr is not None
        assert cr.citation_count == 0
        assert cr.citation_precision is None

    def test_last_citation_result_updated_per_call(self):
        base = _FixedGenerator(answer="")
        wrapper = CitationConstrainedGenerator(base_generator=base, scorer=None)
        contexts = [_make_doc("text")]

        base._answer = "A [1]."
        wrapper.generate("q1", contexts)
        first = wrapper.last_citation_result

        base._answer = "B [1] [2]."
        contexts2 = [_make_doc("t1"), _make_doc("t2")]
        wrapper.generate("q2", contexts2)
        second = wrapper.last_citation_result

        assert first is not None and first.citation_count == 1
        assert second is not None and second.citation_count == 2

    def test_generator_without_system_prompt_attribute(self):
        """ExtractiveGenerator has no system_prompt — wrapper should not raise."""
        from src.generation.extractive import ExtractiveGenerator

        base = ExtractiveGenerator()
        # Should not raise
        wrapper = CitationConstrainedGenerator(base_generator=base, scorer=None)
        contexts = [_make_doc("Some text.")]
        result = wrapper.generate("Q?", contexts)
        assert isinstance(result.text, str)
