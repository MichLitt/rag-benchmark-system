import json
import unittest
from unittest.mock import patch

from src.query.hyde import HOTPOT_HYDE_USER_PROMPT_TEMPLATE, HyDEExpander


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class HyDEExpanderTests(unittest.TestCase):
    def _run_with_payload(
        self,
        payload: dict,
        *,
        model: str = "MiniMax-M2.5",
        max_completion_tokens: int = 128,
    ) -> tuple[str, dict]:
        captured: dict = {}

        def fake_urlopen(request, timeout=0):
            captured["request"] = request
            captured["timeout"] = timeout
            return _DummyResponse(payload)

        expander = HyDEExpander(
            model=model,
            api_key="test-key",
            api_base="https://example.com/v1",
            max_completion_tokens=max_completion_tokens,
        )

        with patch("src.query.hyde.urllib.request.urlopen", side_effect=fake_urlopen):
            expanded = expander.expand("Where is the Eiffel Tower located?")

        return expanded, captured

    def test_expand_returns_non_empty_query(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "The Eiffel Tower is located in Paris, France, on the Champ de Mars.",
                        "role": "assistant",
                    }
                }
            ]
        }

        expanded, _ = self._run_with_payload(payload)

        self.assertIn("Paris", expanded)

    def test_expand_strips_think_tags(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "<think>hidden</think> The Eiffel Tower is in Paris, France.",
                        "role": "assistant",
                    }
                }
            ]
        }

        expanded, _ = self._run_with_payload(payload)

        self.assertEqual(expanded, "The Eiffel Tower is in Paris, France.")

    def test_minimax_request_uses_max_completion_tokens_and_reasoning_split_false(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "The Eiffel Tower is in Paris, France.",
                        "role": "assistant",
                    }
                }
            ]
        }

        _, captured = self._run_with_payload(payload, model="MiniMax-M2.5", max_completion_tokens=96)
        body = json.loads(captured["request"].data.decode("utf-8"))

        self.assertEqual(body["max_completion_tokens"], 96)
        self.assertNotIn("max_tokens", body)
        self.assertFalse(body["reasoning_split"])

    def test_expand_rejects_empty_response(self):
        payload = {"choices": [{"message": {"content": "", "role": "assistant"}}]}

        with self.assertRaisesRegex(RuntimeError, "did not contain a retrievable query"):
            self._run_with_payload(payload)

    def test_custom_prompt_template_is_used(self):
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Bridge entity text",
                        "role": "assistant",
                    }
                }
            ]
        }

        _, captured = self._run_with_payload(
            payload,
            max_completion_tokens=96,
        )
        default_body = json.loads(captured["request"].data.decode("utf-8"))

        captured_hotpot: dict = {}

        def fake_urlopen(request, timeout=0):
            captured_hotpot["request"] = request
            return _DummyResponse(payload)

        expander = HyDEExpander(
            model="MiniMax-M2.5",
            api_key="test-key",
            api_base="https://example.com/v1",
            max_completion_tokens=96,
            user_prompt_template=HOTPOT_HYDE_USER_PROMPT_TEMPLATE,
        )
        with patch("src.query.hyde.urllib.request.urlopen", side_effect=fake_urlopen):
            expander.expand("Who influenced whom?")

        hotpot_body = json.loads(captured_hotpot["request"].data.decode("utf-8"))
        self.assertNotEqual(default_body["messages"][1]["content"], hotpot_body["messages"][1]["content"])
        self.assertIn("both supporting pages", hotpot_body["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
