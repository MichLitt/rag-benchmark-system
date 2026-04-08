"""Unit tests for src/generation/prompts.py."""
import pytest

from src.generation.prompts import (
    CITATION_INSTRUCTION,
    DEFAULT_SYSTEM_PROMPT,
    HOTPOTQA_SYSTEM_PROMPT,
    NQ_SYSTEM_PROMPT,
    TRIVIAQA_SYSTEM_PROMPT,
    resolve_system_prompt,
)


class TestResolveSystemPrompt:
    # --- Priority 1: explicit base_prompt ---

    def test_explicit_base_prompt_wins_over_dataset(self):
        custom = "Custom prompt."
        result = resolve_system_prompt(dataset="hotpotqa", base_prompt=custom)
        assert result == custom

    def test_explicit_base_prompt_wins_over_default(self):
        custom = "Another custom prompt."
        result = resolve_system_prompt(base_prompt=custom)
        assert result == custom

    def test_empty_base_prompt_falls_through_to_dataset(self):
        result = resolve_system_prompt(dataset="triviaqa", base_prompt="")
        assert result == TRIVIAQA_SYSTEM_PROMPT

    def test_whitespace_only_base_prompt_falls_through(self):
        result = resolve_system_prompt(dataset="nq", base_prompt="   ")
        assert result == NQ_SYSTEM_PROMPT

    # --- Priority 2: dataset registry ---

    def test_hotpotqa_dataset_routing(self):
        result = resolve_system_prompt(dataset="hotpotqa")
        assert result == HOTPOTQA_SYSTEM_PROMPT

    def test_triviaqa_dataset_routing(self):
        result = resolve_system_prompt(dataset="triviaqa")
        assert result == TRIVIAQA_SYSTEM_PROMPT

    def test_nq_dataset_routing(self):
        result = resolve_system_prompt(dataset="nq")
        assert result == NQ_SYSTEM_PROMPT

    def test_dataset_name_case_insensitive(self):
        assert resolve_system_prompt(dataset="HotpotQA") == HOTPOTQA_SYSTEM_PROMPT
        assert resolve_system_prompt(dataset="TRIVIAQA") == TRIVIAQA_SYSTEM_PROMPT
        assert resolve_system_prompt(dataset="NQ") == NQ_SYSTEM_PROMPT

    # --- Priority 3: default fallback ---

    def test_no_args_returns_default(self):
        result = resolve_system_prompt()
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_none_dataset_returns_default(self):
        result = resolve_system_prompt(dataset=None)
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_unknown_dataset_returns_default(self):
        result = resolve_system_prompt(dataset="squad")
        assert result == DEFAULT_SYSTEM_PROMPT

    def test_empty_dataset_returns_default(self):
        result = resolve_system_prompt(dataset="")
        assert result == DEFAULT_SYSTEM_PROMPT

    # --- Citation instruction ---

    def test_citation_instruction_appended_to_default(self):
        result = resolve_system_prompt(add_citation_instruction=True)
        assert result == DEFAULT_SYSTEM_PROMPT + CITATION_INSTRUCTION

    def test_citation_instruction_appended_to_dataset_prompt(self):
        result = resolve_system_prompt(dataset="hotpotqa", add_citation_instruction=True)
        assert result == HOTPOTQA_SYSTEM_PROMPT + CITATION_INSTRUCTION

    def test_citation_instruction_appended_to_explicit_prompt(self):
        custom = "Custom."
        result = resolve_system_prompt(base_prompt=custom, add_citation_instruction=True)
        assert result == custom + CITATION_INSTRUCTION

    def test_no_citation_instruction_by_default(self):
        result = resolve_system_prompt(dataset="nq")
        assert CITATION_INSTRUCTION not in result

    # --- Prompt content sanity checks ---

    def test_hotpotqa_prompt_contains_final_answer_format(self):
        assert "Final answer:" in HOTPOTQA_SYSTEM_PROMPT

    def test_triviaqa_prompt_constrains_length(self):
        assert "3 words" in TRIVIAQA_SYSTEM_PROMPT or "≤3" in TRIVIAQA_SYSTEM_PROMPT

    def test_citation_instruction_contains_example(self):
        assert "[1]" in CITATION_INSTRUCTION
        assert "[2, 3]" in CITATION_INSTRUCTION
