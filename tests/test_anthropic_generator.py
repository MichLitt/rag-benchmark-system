"""Tests for AnthropicCompatibleGenerator and factory routing.

All tests mock the anthropic SDK so no real API calls are made.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from src.generation.anthropic_compatible import AnthropicCompatibleGenerator
from src.generation.factory import build_generator
from src.generation.openai_compatible import DEFAULT_SYSTEM_PROMPT
from src.types import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str = "Paris is the capital of France.", title: str = "France") -> Document:
    return Document(doc_id="d1", text=text, title=title)


def _make_response(text: str, input_tokens: int = 10, output_tokens: int = 5) -> MagicMock:
    """Build a fake anthropic.types.Message-like object."""
    content_block = MagicMock()
    content_block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


def _make_generator(
    *,
    api_base: str | None = None,
    max_output_tokens: int = 128,
    input_price_per_1m: float = 0.0,
    output_price_per_1m: float = 0.0,
) -> AnthropicCompatibleGenerator:
    with patch("anthropic.Anthropic"):
        return AnthropicCompatibleGenerator(
            model="minimax-m2.7",
            api_key="fake-key",
            api_base=api_base,
            max_output_tokens=max_output_tokens,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
        )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestAnthropicGeneratorConstructor(unittest.TestCase):

    def test_no_api_base_omits_base_url(self):
        with patch("anthropic.Anthropic") as mock_cls:
            AnthropicCompatibleGenerator(model="m", api_key="k")
            call_kwargs = mock_cls.call_args.kwargs
            self.assertNotIn("base_url", call_kwargs)

    def test_api_base_passed_as_base_url(self):
        with patch("anthropic.Anthropic") as mock_cls:
            AnthropicCompatibleGenerator(model="m", api_key="k", api_base="https://custom.api/v1/")
            call_kwargs = mock_cls.call_args.kwargs
            # trailing slash stripped
            self.assertEqual(call_kwargs["base_url"], "https://custom.api/v1")

    def test_empty_string_api_base_treated_as_none(self):
        with patch("anthropic.Anthropic") as mock_cls:
            AnthropicCompatibleGenerator(model="m", api_key="k", api_base="")
            call_kwargs = mock_cls.call_args.kwargs
            self.assertNotIn("base_url", call_kwargs)

    def test_timeout_passed_to_client(self):
        with patch("anthropic.Anthropic") as mock_cls:
            AnthropicCompatibleGenerator(model="m", api_key="k", timeout_sec=30)
            call_kwargs = mock_cls.call_args.kwargs
            self.assertEqual(call_kwargs["timeout"], 30.0)

    def test_invalid_max_output_tokens_raises(self):
        with patch("anthropic.Anthropic"):
            with self.assertRaises(ValueError, msg="max_output_tokens must be > 0"):
                AnthropicCompatibleGenerator(model="m", api_key="k", max_output_tokens=0)

    def test_default_system_prompt_used_when_blank(self):
        with patch("anthropic.Anthropic"):
            gen = AnthropicCompatibleGenerator(model="m", api_key="k", system_prompt="   ")
            self.assertEqual(gen.system_prompt, DEFAULT_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# generate() — empty contexts
# ---------------------------------------------------------------------------

class TestAnthropicGenerateEmpty(unittest.TestCase):

    def test_empty_contexts_returns_do_not_know(self):
        gen = _make_generator()
        result = gen.generate("What is X?", [])
        self.assertEqual(result.text, "I do not know.")
        self.assertEqual(result.provider, "anthropic")
        self.assertEqual(result.model, "minimax-m2.7")

    def test_empty_contexts_no_api_call(self):
        with patch("anthropic.Anthropic") as mock_cls:
            gen = AnthropicCompatibleGenerator(model="m", api_key="k")
            gen.generate("q", [])
            # messages.create should NOT have been called
            mock_cls.return_value.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# generate() — normal response
# ---------------------------------------------------------------------------

class TestAnthropicGenerateNormal(unittest.TestCase):

    def test_text_extracted_from_first_content_block(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response("  Paris  ")
        result = gen.generate("Capital of France?", [_doc()])
        self.assertEqual(result.text, "Paris")

    def test_provider_is_anthropic(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response("Paris")
        result = gen.generate("q", [_doc()])
        self.assertEqual(result.provider, "anthropic")

    def test_model_propagated(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response("Paris")
        result = gen.generate("q", [_doc()])
        self.assertEqual(result.model, "minimax-m2.7")

    def test_token_counts_from_usage(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response(
            "Paris", input_tokens=42, output_tokens=7
        )
        result = gen.generate("q", [_doc()])
        self.assertEqual(result.input_tokens, 42)
        self.assertEqual(result.output_tokens, 7)

    def test_no_reasoning_tokens_field(self):
        """Anthropic generator never sets reasoning_tokens."""
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response("Paris")
        result = gen.generate("q", [_doc()])
        self.assertIsNone(result.reasoning_tokens)

    def test_messages_create_called_with_correct_args(self):
        gen = _make_generator(max_output_tokens=64)
        gen._client.messages.create.return_value = _make_response("Paris")
        gen.generate("Capital?", [_doc()])
        call_kwargs = gen._client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "minimax-m2.7")
        self.assertEqual(call_kwargs["max_tokens"], 64)
        self.assertEqual(call_kwargs["system"], DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")

    def test_user_prompt_contains_question_and_context(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response("Paris")
        gen.generate("Capital of France?", [_doc("Paris is the capital.", "France")])
        call_kwargs = gen._client.messages.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        self.assertIn("Capital of France?", user_content)
        self.assertIn("Paris is the capital.", user_content)
        self.assertIn("France", user_content)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

class TestAnthropicCost(unittest.TestCase):

    def test_cost_zero_when_prices_not_set(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response(
            "Paris", input_tokens=100, output_tokens=20
        )
        result = gen.generate("q", [_doc()])
        self.assertIsNone(result.cost_usd)

    def test_cost_computed_correctly(self):
        gen = _make_generator(input_price_per_1m=1.0, output_price_per_1m=5.0)
        gen._client.messages.create.return_value = _make_response(
            "Paris", input_tokens=1_000_000, output_tokens=1_000_000
        )
        result = gen.generate("q", [_doc()])
        assert result.cost_usd is not None
        self.assertAlmostEqual(result.cost_usd, 6.0, places=6)

    def test_cost_none_when_tokens_missing(self):
        gen = _make_generator(input_price_per_1m=1.0, output_price_per_1m=1.0)
        resp = _make_response("Paris")
        resp.usage.input_tokens = None
        resp.usage.output_tokens = None
        gen._client.messages.create.return_value = resp
        result = gen.generate("q", [_doc()])
        self.assertIsNone(result.cost_usd)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestAnthropicErrors(unittest.TestCase):

    def _api_status_error(self):
        import anthropic
        err = MagicMock(spec=anthropic.APIStatusError)
        err.status_code = 429
        err.message = "rate limit"
        return err

    def test_api_status_error_raises_runtime_error(self):
        import anthropic
        gen = _make_generator()
        gen._client.messages.create.side_effect = anthropic.APIStatusError(
            "rate limit", response=MagicMock(), body={}
        )
        with self.assertRaises(RuntimeError, msg="429"):
            gen.generate("q", [_doc()])

    def test_api_connection_error_raises_runtime_error(self):
        import anthropic
        gen = _make_generator()
        gen._client.messages.create.side_effect = anthropic.APIConnectionError(request=MagicMock())
        with self.assertRaises(RuntimeError):
            gen.generate("q", [_doc()])

    def test_empty_content_blocks_raises_runtime_error(self):
        gen = _make_generator()
        resp = _make_response("Paris")
        resp.content = []
        gen._client.messages.create.return_value = resp
        with self.assertRaises(RuntimeError, msg="content blocks"):
            gen.generate("q", [_doc()])

    def test_blank_text_raises_runtime_error(self):
        gen = _make_generator()
        gen._client.messages.create.return_value = _make_response("   ")
        with self.assertRaises(RuntimeError, msg="final answer"):
            gen.generate("q", [_doc()])


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

class TestFactory(unittest.TestCase):

    def _env(self):
        return {
            "MINIMAX_API_KEY": "fake-key",
            "MINIMAX_BASE_URL": "https://api.minimax.io/anthropic",
        }

    def _cfg(self, mode: str) -> dict:
        return {
            "generation": {
                "mode": mode,
                "model": "minimax-m2.7",
                "api_key_env": "MINIMAX_API_KEY",
                "api_base_env": "MINIMAX_BASE_URL",
            }
        }

    def test_factory_routes_anthropic_mode(self):
        with patch("anthropic.Anthropic"), patch.dict(os.environ, self._env()):
            gen = build_generator(self._cfg("anthropic"))
        self.assertIsInstance(gen, AnthropicCompatibleGenerator)

    def test_factory_routes_anthropic_compatible_mode(self):
        with patch("anthropic.Anthropic"), patch.dict(os.environ, self._env()):
            gen = build_generator(self._cfg("anthropic_compatible"))
        self.assertIsInstance(gen, AnthropicCompatibleGenerator)

    def test_factory_anthropic_missing_key_raises(self):
        with patch("anthropic.Anthropic"), patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError, msg="ANTHROPIC_API_KEY"):
                build_generator(self._cfg("anthropic"))

    def test_factory_anthropic_missing_model_raises(self):
        cfg = {
            "generation": {
                "mode": "anthropic",
                "model": "",
                "api_key_env": "MINIMAX_API_KEY",
            }
        }
        with patch("anthropic.Anthropic"), patch.dict(os.environ, {"MINIMAX_API_KEY": "k"}):
            with self.assertRaises(ValueError, msg="model"):
                build_generator(cfg)

    def test_factory_anthropic_api_base_from_env(self):
        with patch("anthropic.Anthropic") as mock_cls, patch.dict(os.environ, self._env()):
            build_generator(self._cfg("anthropic"))
            call_kwargs = mock_cls.call_args.kwargs
            self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/anthropic")

    def test_factory_openai_compatible_still_works(self):
        from src.generation.openai_compatible import OpenAICompatibleGenerator
        env = {"GLM_API_KEY": "fake", "GLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4"}
        cfg = {
            "generation": {
                "mode": "openai_compatible",
                "model": "glm-5",
                "api_key_env": "GLM_API_KEY",
                "api_base_env": "GLM_BASE_URL",
            }
        }
        with patch.dict(os.environ, env):
            gen = build_generator(cfg)
        self.assertIsInstance(gen, OpenAICompatibleGenerator)
