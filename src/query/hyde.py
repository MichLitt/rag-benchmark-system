from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from src.generation.openai_compatible import (
    _extract_text,
    _is_minimax_model,
    _strip_think_blocks,
)
from src.logging_utils import get_logger

logger = get_logger(__name__)


DEFAULT_HYDE_SYSTEM_PROMPT = (
    "Write one short factual paragraph for dense retrieval. "
    "Output only the paragraph. "
    'Do not say "I do not know." '
    "Do not reveal reasoning. Do not output <think> tags. "
    "Include key entities, likely answer terms, and supporting facts."
)
DEFAULT_HYDE_USER_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Write a hypothetical answer paragraph with the key entities and facts."
)

HOTPOT_HYDE_SYSTEM_PROMPT = (
    "Write one short multi-hop evidence paragraph for dense retrieval. "
    "Output only the paragraph. "
    'Do not say "I do not know." '
    "Do not reveal reasoning. Do not output <think> tags. "
    "Mention the likely bridge entity, the answer-side entity, and relation clues that connect them."
)
HOTPOT_HYDE_USER_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Write a hypothetical evidence paragraph that would help retrieve both supporting pages "
    "for this multi-hop question."
)


class HyDEExpander:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 256,
        timeout_sec: int = 60,
        system_prompt: str = DEFAULT_HYDE_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_HYDE_USER_PROMPT_TEMPLATE,
    ) -> None:
        self.model = model.strip()
        if not self.model:
            raise ValueError("HyDE model must be non-empty.")
        self.api_key = api_key.strip()
        if not self.api_key:
            raise ValueError("HyDE api_key must be non-empty.")
        self.api_base = api_base.rstrip("/")
        if not self.api_base:
            raise ValueError("HyDE api_base must be non-empty.")
        self.temperature = float(temperature)
        self.max_completion_tokens = int(max_completion_tokens)
        if self.max_completion_tokens <= 0:
            raise ValueError(
                f"HyDE max_completion_tokens must be > 0, got {self.max_completion_tokens}"
            )
        self.timeout_sec = int(timeout_sec)
        self.system_prompt = system_prompt.strip() or DEFAULT_HYDE_SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template.strip() or DEFAULT_HYDE_USER_PROMPT_TEMPLATE

    def _request_token_budget(self) -> tuple[str, int]:
        if _is_minimax_model(self.model):
            return "max_completion_tokens", self.max_completion_tokens
        return "max_tokens", self.max_completion_tokens

    def _build_request_body(self, question: str) -> dict[str, Any]:
        budget_key, budget_value = self._request_token_budget()
        user_prompt = self.user_prompt_template.format(question=question.strip())
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "reasoning_split": False,
        }
        body[budget_key] = budget_value
        return body

    def expand(self, question: str) -> str:
        body = self._build_request_body(question)
        request = urllib.request.Request(
            url=f"{self.api_base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            logger.warning("HyDE HTTP error %s: %.200s", exc.code, detail)
            raise RuntimeError(f"HyDE request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            logger.warning("HyDE URL error: %s", exc.reason)
            raise RuntimeError(f"HyDE request failed: {exc.reason}") from exc

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("HyDE response did not contain any choices.")

        message = choices[0].get("message", {})
        expanded = _strip_think_blocks(_extract_text(message.get("content")))
        if not expanded:
            raise RuntimeError("HyDE response did not contain a retrievable query.")
        if expanded.lower() == "i do not know.":
            raise RuntimeError('HyDE response returned the forbidden fallback "I do not know."')
        return expanded

    def expand_queries(self, question: str) -> list[str]:
        return [self.expand(question)]
