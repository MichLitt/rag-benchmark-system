"""Unit tests for src/generation/postprocess.py."""
import pytest

from src.generation.postprocess import postprocess_answer, strip_hedging


class TestStripHedging:
    # --- Hedge patterns that should be stripped ---

    def test_according_to_context(self):
        text = "According to the context, the answer is Paris."
        result, hedged = strip_hedging(text)
        assert result == "the answer is Paris."
        assert hedged is True

    def test_according_to_passage(self):
        text = "According to the passage, Lincoln was president."
        result, hedged = strip_hedging(text)
        assert result == "Lincoln was president."
        assert hedged is True

    def test_according_to_retrieved_passages(self):
        text = "According to the retrieved passages, the capital is Tokyo."
        result, hedged = strip_hedging(text)
        assert result == "the capital is Tokyo."
        assert hedged is True

    def test_based_on_context(self):
        text = "Based on the context, the film was released in 1994."
        result, hedged = strip_hedging(text)
        assert result == "the film was released in 1994."
        assert hedged is True

    def test_based_on_retrieved_context(self):
        text = "Based on the retrieved context, the answer is 42."
        result, hedged = strip_hedging(text)
        assert result == "the answer is 42."
        assert hedged is True

    def test_the_context_states_that(self):
        text = "The context states that the river is 6,650 km long."
        result, hedged = strip_hedging(text)
        assert result == "the river is 6,650 km long."
        assert hedged is True

    def test_the_passage_mentions(self):
        text = "The passage mentions that Shakespeare was born in 1564."
        result, hedged = strip_hedging(text)
        assert result == "Shakespeare was born in 1564."
        assert hedged is True

    def test_as_mentioned_in_context(self):
        text = "As mentioned in the context, Berlin is the capital."
        result, hedged = strip_hedging(text)
        assert result == "Berlin is the capital."
        assert hedged is True

    def test_as_stated_in_the_passage(self):
        text = "As stated in the passage, the speed is 299,792 km/s."
        result, hedged = strip_hedging(text)
        assert result == "the speed is 299,792 km/s."
        assert hedged is True

    def test_the_answer_is(self):
        text = "The answer is Mount Everest."
        result, hedged = strip_hedging(text)
        assert result == "Mount Everest."
        assert hedged is True

    def test_the_answer_is_with_dash(self):
        text = "The answer is - Marie Curie"
        result, hedged = strip_hedging(text)
        assert result == "Marie Curie"
        assert hedged is True

    def test_from_the_context(self):
        text = "From the context, we can see the author is Tolkien."
        result, hedged = strip_hedging(text)
        assert result == "we can see the author is Tolkien."
        assert hedged is True

    def test_final_answer_colon(self):
        """HotpotQA dataset-specific prompt format."""
        text = "Final answer: Paris"
        result, hedged = strip_hedging(text)
        assert result == "Paris"
        assert hedged is True

    def test_final_answer_dash(self):
        text = "Final Answer - 1994"
        result, hedged = strip_hedging(text)
        assert result == "1994"
        assert hedged is True

    def test_case_insensitive(self):
        text = "ACCORDING TO THE CONTEXT, the answer is yes."
        result, hedged = strip_hedging(text)
        assert result == "the answer is yes."
        assert hedged is True

    # --- Patterns that should NOT be stripped ---

    def test_no_hedge_plain_answer(self):
        text = "Paris"
        result, hedged = strip_hedging(text)
        assert result == "Paris"
        assert hedged is False

    def test_no_hedge_sentence(self):
        text = "The Eiffel Tower is located in Paris."
        result, hedged = strip_hedging(text)
        assert result == "The Eiffel Tower is located in Paris."
        assert hedged is False

    def test_no_hedge_number(self):
        text = "42"
        result, hedged = strip_hedging(text)
        assert result == "42"
        assert hedged is False

    def test_no_hedge_yes_no(self):
        text = "Yes"
        result, hedged = strip_hedging(text)
        assert result == "Yes"
        assert hedged is False

    # --- Edge cases ---

    def test_stripping_would_leave_empty_returns_original(self):
        """If stripping produces empty string, return original."""
        text = "According to the context,"
        result, hedged = strip_hedging(text)
        # was_hedged=True because pattern matched, but text preserved
        assert result == text.strip()
        assert hedged is True

    def test_leading_whitespace_stripped(self):
        text = "  Paris  "
        result, hedged = strip_hedging(text)
        assert result == "Paris"
        assert hedged is False

    def test_only_first_pattern_applies(self):
        """Double-hedged text should only strip the first prefix."""
        text = "According to the context, based on the passage, the answer is Rome."
        result, hedged = strip_hedging(text)
        # "according to the context," is stripped; "based on the passage," remains
        assert result == "based on the passage, the answer is Rome."
        assert hedged is True


class TestPostprocessAnswer:
    """postprocess_answer is a thin wrapper — verify it delegates correctly."""

    def test_delegates_to_strip_hedging(self):
        text = "Based on the context, the answer is Neptune."
        result, hedged = postprocess_answer(text)
        assert result == "the answer is Neptune."
        assert hedged is True

    def test_plain_answer_unchanged(self):
        result, hedged = postprocess_answer("Neptune")
        assert result == "Neptune"
        assert hedged is False
