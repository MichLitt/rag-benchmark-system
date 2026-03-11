import os
import unittest
from unittest.mock import patch

from src.generation.factory import build_generator
from src.generation.openai_compatible import OpenAICompatibleGenerator


class GenerationFactoryTests(unittest.TestCase):
    def test_factory_auto_enables_reasoning_split_for_minimax(self):
        cfg = {
            "generation": {
                "mode": "llm",
                "model": "MiniMax-M2.5",
                "api_base": "https://example.com/v1",
                "api_key_env": "LLM_API_KEY",
                "reasoning_split": "auto",
                "max_output_tokens": 128,
            }
        }

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            generator = build_generator(cfg)

        self.assertIsInstance(generator, OpenAICompatibleGenerator)
        self.assertTrue(generator.reasoning_split)
        self.assertEqual(generator.max_completion_tokens, 512)

    def test_factory_prefers_explicit_max_completion_tokens(self):
        cfg = {
            "generation": {
                "mode": "llm",
                "model": "MiniMax-M2.5",
                "api_base": "https://example.com/v1",
                "api_key_env": "LLM_API_KEY",
                "reasoning_split": True,
                "max_output_tokens": 64,
                "max_completion_tokens": 1024,
            }
        }

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            generator = build_generator(cfg)

        self.assertEqual(generator.max_completion_tokens, 1024)

    def test_factory_keeps_auto_reasoning_split_off_for_non_minimax(self):
        cfg = {
            "generation": {
                "mode": "llm",
                "model": "gpt-4o-mini",
                "api_base": "https://example.com/v1",
                "api_key_env": "LLM_API_KEY",
                "reasoning_split": "auto",
                "max_output_tokens": 128,
            }
        }

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            generator = build_generator(cfg)

        self.assertFalse(generator.reasoning_split)
        self.assertIsNone(generator.max_completion_tokens)


if __name__ == "__main__":
    unittest.main()
