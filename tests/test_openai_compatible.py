import json
import unittest
from unittest.mock import patch

from src.generation.openai_compatible import OpenAICompatibleGenerator
from src.types import Document


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class OpenAICompatibleGeneratorTests(unittest.TestCase):
    def _docs(self) -> list[Document]:
        return [Document(doc_id="d1", title="France", text="Paris is the capital of France.")]

    def _run_with_payload(
        self,
        payload: dict,
        *,
        model: str = "MiniMax-M2.5",
        max_output_tokens: int = 128,
        max_completion_tokens: int | None = 512,
        reasoning_split: bool | str | None = True,
    ):
        captured: dict = {}

        def fake_urlopen(request, timeout=0):
            captured["request"] = request
            captured["timeout"] = timeout
            return _DummyResponse(payload)

        generator = OpenAICompatibleGenerator(
            model=model,
            api_key="test-key",
            api_base="https://example.com/v1",
            max_output_tokens=max_output_tokens,
            max_completion_tokens=max_completion_tokens,
            reasoning_split=reasoning_split,
        )

        with patch("src.generation.openai_compatible.urllib.request.urlopen", side_effect=fake_urlopen):
            result = generator.generate("What is the capital of France?", self._docs())

        return result, captured

    def test_generate_returns_plain_content(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Paris", "role": "assistant"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

        result, _ = self._run_with_payload(payload)

        self.assertEqual(result.text, "Paris")
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 2)
        self.assertIsNone(result.reasoning_tokens)

    def test_generate_strips_think_tags_and_keeps_final_answer(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "<think>hidden reasoning</think>\nParis",
                        "role": "assistant",
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

        result, _ = self._run_with_payload(payload)

        self.assertEqual(result.text, "Paris")

    def test_generate_reads_reasoning_details_and_final_answer(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "Paris",
                        "role": "assistant",
                        "reasoning_details": [{"text": "reasoning"}],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 27,
                "completion_tokens": 55,
                "completion_tokens_details": {"reasoning_tokens": 53},
            },
        }

        result, _ = self._run_with_payload(payload)

        self.assertEqual(result.text, "Paris")
        self.assertEqual(result.reasoning_tokens, 53)

    def test_generate_raises_on_reasoning_only_length_cutoff(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "<think>reasoning only</think>",
                        "role": "assistant",
                        "reasoning_details": [{"text": "reasoning"}],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 27,
                "completion_tokens": 128,
                "completion_tokens_details": {"reasoning_tokens": 128},
            },
        }

        with self.assertRaisesRegex(RuntimeError, "Increase generation.max_completion_tokens"):
            self._run_with_payload(payload, max_completion_tokens=128)

    def test_minimax_request_prefers_max_completion_tokens(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Paris", "role": "assistant"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

        _, captured = self._run_with_payload(
            payload,
            model="MiniMax-M2.5",
            max_output_tokens=64,
            max_completion_tokens=512,
            reasoning_split="auto",
        )
        body = json.loads(captured["request"].data.decode("utf-8"))

        self.assertEqual(body["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", body)
        self.assertTrue(body["reasoning_split"])

    def test_non_minimax_request_uses_legacy_max_tokens(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "Paris", "role": "assistant"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

        _, captured = self._run_with_payload(
            payload,
            model="gpt-4o-mini",
            max_output_tokens=64,
            max_completion_tokens=256,
            reasoning_split="auto",
        )
        body = json.loads(captured["request"].data.decode("utf-8"))

        self.assertEqual(body["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", body)
        self.assertNotIn("reasoning_split", body)


if __name__ == "__main__":
    unittest.main()
