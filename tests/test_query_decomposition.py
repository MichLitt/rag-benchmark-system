import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.query.decomposition import HotpotDecomposeExpander


def _load_script_module(script_name: str):
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class HotpotDecomposeExpanderTests(unittest.TestCase):
    def _run_with_payload(
        self,
        payload: dict,
        *,
        include_original_query: bool = True,
        cache_dir: Path | None = None,
    ) -> tuple[list[str], dict, HotpotDecomposeExpander]:
        captured: dict = {}

        def fake_urlopen(request, timeout=0):
            captured["request"] = request
            captured["timeout"] = timeout
            return _DummyResponse(payload)

        expander = HotpotDecomposeExpander(
            model="MiniMax-M2.5",
            api_key="test-key",
            api_base="https://example.com/v1",
            include_original_query=include_original_query,
            max_queries=3,
            cache_dir=cache_dir,
        )

        with patch("src.query.decomposition.urllib.request.urlopen", side_effect=fake_urlopen):
            queries = expander.expand_queries(
                "Which actor starred in the film directed by Tim Burton about Ed Wood?"
            )

        return queries, captured, expander

    def test_expand_queries_parses_json_queries(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"queries": ["Tim Burton Ed Wood film cast", '
                            '"Ed Wood film actor Johnny Depp"]}'
                        ),
                        "role": "assistant",
                    }
                }
            ]
        }

        queries, _, expander = self._run_with_payload(payload)

        self.assertEqual(
            queries,
            [
                "Which actor starred in the film directed by Tim Burton about Ed Wood?",
                "Tim Burton Ed Wood film cast",
                "Ed Wood film actor Johnny Depp",
            ],
        )
        self.assertEqual(expander.get_last_expansion_metadata()["failure_reason"], "")

    def test_expand_queries_keeps_trailing_punctuation_candidates(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "- Tim Burton Ed Wood cast?\n- Ed Wood lead actor!",
                        "role": "assistant",
                    }
                }
            ]
        }

        queries, _, _ = self._run_with_payload(payload)

        self.assertIn("Tim Burton Ed Wood cast", queries)
        self.assertIn("Ed Wood lead actor", queries)

    def test_expand_queries_truncates_when_all_candidates_are_over_length(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"queries": ["Tim Burton Ed Wood film cast lead actor Johnny Depp role", '
                            '"Ed Wood movie actor main cast lead performer biography profile"]}'
                        ),
                        "role": "assistant",
                    }
                }
            ]
        }

        queries, _, _ = self._run_with_payload(payload, include_original_query=False)

        self.assertEqual(len(queries), 2)
        self.assertLessEqual(max(len(query.split()) for query in queries), 12)

    def test_expand_queries_salvages_quoted_queries_from_think_only_output(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<think>\n"
                            "Use queries:\n"
                            "\"Scott Derrickson nationality\"\n"
                            "\"Ed Wood director nationality\"\n"
                            "</think>"
                        ),
                        "role": "assistant",
                    }
                }
            ]
        }

        queries, _, _ = self._run_with_payload(payload)

        self.assertIn("Scott Derrickson nationality", queries)
        self.assertIn("Ed Wood director nationality", queries)

    def test_expand_queries_uses_question_fallback_and_writes_failure_diagnostics(self):
        payload = {"choices": [{"message": {"content": "", "role": "assistant"}}]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            queries, _, expander = self._run_with_payload(
                payload,
                include_original_query=False,
                cache_dir=Path(tmp_dir),
            )
            metadata = expander.get_last_expansion_metadata()
            failure_path = Path(tmp_dir) / "failures" / f"{metadata['cache_key']}.json"
            self.assertTrue(failure_path.exists())
            saved = json.loads(failure_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["failure_reason"], "empty_content")

        self.assertEqual(
            queries,
            ["Which actor starred in the film directed by Tim Burton about Ed Wood?"],
        )
        self.assertEqual(metadata["failure_reason"], "empty_content")
        self.assertTrue(metadata["used_fallback"])

    def test_expand_queries_uses_cache_after_first_request(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"queries": ["Tim Burton Ed Wood cast", "Ed Wood lead actor"]}',
                        "role": "assistant",
                    }
                }
            ]
        }
        calls = {"count": 0}

        def fake_urlopen(request, timeout=0):
            calls["count"] += 1
            return _DummyResponse(payload)

        with tempfile.TemporaryDirectory() as tmp_dir:
            expander = HotpotDecomposeExpander(
                model="MiniMax-M2.5",
                api_key="test-key",
                api_base="https://example.com/v1",
                include_original_query=True,
                max_queries=3,
                cache_dir=Path(tmp_dir),
            )

            with patch("src.query.decomposition.urllib.request.urlopen", side_effect=fake_urlopen):
                first = expander.expand_queries("Who starred in Ed Wood?")
                second = expander.expand_queries("Who starred in Ed Wood?")

        self.assertEqual(calls["count"], 1)
        self.assertEqual(first, second)
        self.assertTrue(expander.get_last_expansion_metadata()["cache_hit"])

    def test_replay_script_writes_summary_and_results(self):
        script = _load_script_module("replay_hotpot_decompose_failures.py")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            qa_path = root / "qa.jsonl"
            predictions_path = root / "predictions.json"
            config_path = root / "config.yaml"
            output_dir = root / "replay"

            qa_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Who starred in Ed Wood?",
                        "golden_answers": ["Johnny Depp"],
                        "gold_answer": "Johnny Depp",
                        "gold_titles": ["Ed Wood", "Johnny Depp"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            predictions_path.write_text(
                json.dumps(
                    [
                        {
                            "query_id": "q1",
                            "query_expansion_error": "RuntimeError: Decomposition response did not contain any retrievable queries.",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                "query_expansion:\n"
                "  mode: hotpot_decompose\n"
                "  model: MiniMax-M2.5\n"
                "  api_base: https://example.com/v1\n"
                "  api_key_env: LLM_API_KEY\n"
                "generation:\n"
                "  api_base_env: LLM_BASE_URL\n"
                "  api_key_env: LLM_API_KEY\n",
                encoding="utf-8",
            )

            class _FakeExpander:
                def expand_queries(self, question: str) -> list[str]:
                    return ["Ed Wood cast", "Johnny Depp Ed Wood"]

                def get_last_expansion_metadata(self) -> dict:
                    return {
                        "failure_reason": "",
                        "used_fallback": False,
                        "cache_key": "abc",
                        "salvage_stage": "strict_json",
                    }

            argv = [
                "replay_hotpot_decompose_failures.py",
                "--predictions-path",
                str(predictions_path),
                "--config",
                str(config_path),
                "--qa-path",
                str(qa_path),
                "--output-dir",
                str(output_dir),
            ]
            with patch.object(script, "build_query_expander", return_value=_FakeExpander()):
                with patch("sys.argv", argv):
                    script.main()

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["total_failed_examples"], 1)
            self.assertEqual(summary["usable_query_count"], 1)
            replay_rows = [
                json.loads(line)
                for line in (output_dir / "replay_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(replay_rows[0]["query_id"], "q1")
            self.assertEqual(replay_rows[0]["new_queries"], ["Ed Wood cast", "Johnny Depp Ed Wood"])


if __name__ == "__main__":
    unittest.main()
