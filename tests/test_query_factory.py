import os
import unittest
from unittest.mock import patch

from src.query.factory import build_query_expander, resolve_query_expansion_mode


class QueryFactoryTests(unittest.TestCase):
    def test_resolve_query_expansion_mode_auto_is_dataset_specific(self):
        cfg = {"query_expansion": {"mode": "auto"}}

        self.assertEqual(resolve_query_expansion_mode(cfg, "hotpotqa"), "off")
        self.assertEqual(resolve_query_expansion_mode(cfg, "nq"), "hyde")
        self.assertEqual(resolve_query_expansion_mode(cfg, "triviaqa"), "off")

    def test_resolve_query_expansion_mode_respects_dataset_allowlist(self):
        cfg = {"query_expansion": {"mode": "auto", "datasets": ["hotpotqa", "nq"]}}

        self.assertEqual(resolve_query_expansion_mode(cfg, "triviaqa"), "off")
        self.assertEqual(resolve_query_expansion_mode(cfg, "nq"), "hyde")

    def test_build_query_expander_uses_hotpot_prompt_in_explicit_hotpot_mode(self):
        cfg = {
            "generation": {
                "model": "MiniMax-M2.5",
                "api_base": "https://example.com/v1",
                "api_key_env": "LLM_API_KEY",
            },
            "query_expansion": {
                "mode": "hotpot_hyde",
            },
        }

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://example.com/v1"}, clear=False):
            expander = build_query_expander(cfg, dataset_name="hotpotqa")

        self.assertIsNotNone(expander)
        self.assertIn("multi-hop", expander.system_prompt.lower())
        self.assertIn("supporting pages", expander.user_prompt_template.lower())

    def test_build_query_expander_supports_hotpot_decomposition_mode(self):
        cfg = {
            "generation": {
                "model": "MiniMax-M2.5",
                "api_base": "https://example.com/v1",
                "api_key_env": "LLM_API_KEY",
            },
            "query_expansion": {
                "mode": "hotpot_decompose",
                "include_original_query": True,
                "max_queries": 3,
            },
        }

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key", "LLM_BASE_URL": "https://example.com/v1"}, clear=False):
            expander = build_query_expander(cfg, dataset_name="hotpotqa")

        self.assertIsNotNone(expander)
        self.assertTrue(expander.include_original_query)
        self.assertEqual(expander.max_queries, 3)

    def test_build_query_expander_returns_none_when_dataset_not_selected(self):
        cfg = {
            "generation": {
                "model": "MiniMax-M2.5",
                "api_base": "https://example.com/v1",
                "api_key_env": "LLM_API_KEY",
            },
            "query_expansion": {
                "mode": "auto",
                "datasets": ["hotpotqa", "nq"],
            },
        }

        with patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=False):
            expander = build_query_expander(cfg, dataset_name="triviaqa")

        self.assertIsNone(expander)
