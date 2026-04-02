from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from src.generation.base import GenerationResult
from src.logging_utils import get_logger
from src.types import Document

logger = get_logger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You answer questions based on the retrieved context. "
    "Extract the most relevant information from the context to answer the question. "
    "Keep the answer short and factual — a few words or one sentence. "
    "Do not reveal reasoning. Do not output <think> tags or chain-of-thought. "
    "Output only the final answer."
)


def _is_minimax_model(model: str) -> bool:
    return model.strip().lower().startswith("minimax")


def _coerce_reasoning_split(value: bool | str | None, model: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return _is_minimax_model(model)

    text = str(value).strip().lower()
    if not text or text == "auto":
        return _is_minimax_model(model)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported reasoning_split value: {value}")


def _format_contexts(contexts: list[Document]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(contexts, start=1):
        title = doc.title.strip() or "(untitled)"
        text = doc.text.strip()
        blocks.append(f"[{index}] Title: {title}\nContent: {text}")
    return "\n\n".join(blocks)


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text.strip())
        return "\n".join(part for part in parts if part).strip()
    return ""


def _strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</think>", " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split()).strip()


def _extract_reasoning_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    reasoning_tokens = details.get("reasoning_tokens")
    if isinstance(reasoning_tokens, int):
        return reasoning_tokens
    return None


class OpenAICompatibleGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_output_tokens: int = 128,
        max_completion_tokens: int | None = None,
        reasoning_split: bool | str | None = None,
        timeout_sec: int = 60,
        input_price_per_1m: float = 0.0,
        output_price_per_1m: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be > 0, got {self.max_output_tokens}")
        self.max_completion_tokens = (
            int(max_completion_tokens) if max_completion_tokens is not None else None
        )
        if self.max_completion_tokens is not None and self.max_completion_tokens <= 0:
            raise ValueError(
                f"max_completion_tokens must be > 0, got {self.max_completion_tokens}"
            )
        self.reasoning_split = _coerce_reasoning_split(reasoning_split, self.model)
        self.timeout_sec = int(timeout_sec)
        self.input_price_per_1m = float(input_price_per_1m)
        self.output_price_per_1m = float(output_price_per_1m)

    def _request_token_budget(self) -> tuple[str, int]:
        if _is_minimax_model(self.model) and self.max_completion_tokens is not None:
            return "max_completion_tokens", self.max_completion_tokens
        if self.max_completion_tokens is not None:
            return "max_tokens", self.max_completion_tokens
        return "max_tokens", self.max_output_tokens

    def _build_request_body(self, question: str, contexts: list[Document]) -> dict[str, Any]:
        user_prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Retrieved context:\n{_format_contexts(contexts)}\n\n"
            'Answer using the most relevant parts of the context above. Provide the best factual answer you can.'
        )
        budget_key, budget_value = self._request_token_budget()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        body[budget_key] = budget_value
        if self.reasoning_split:
            body["reasoning_split"] = True
        return body

    def _compute_cost(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        if prompt_tokens is None or completion_tokens is None:
            return None
        if self.input_price_per_1m <= 0.0 and self.output_price_per_1m <= 0.0:
            return None
        return (
            (prompt_tokens * self.input_price_per_1m)
            + (completion_tokens * self.output_price_per_1m)
        ) / 1_000_000.0

    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        if not contexts:
            return GenerationResult(
                text="I do not know.",
                provider="openai_compatible",
                model=self.model,
            )

        body = self._build_request_body(question, contexts)
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
            logger.warning("LLM HTTP error %s: %.200s", exc.code, detail)
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            logger.warning("LLM URL error: %s", exc.reason)
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response did not contain any choices.")

        message = choices[0].get("message", {})
        raw_text = _extract_text(message.get("content"))
        text = _strip_think_blocks(raw_text)
        finish_reason = choices[0].get("finish_reason")
        usage = payload.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        reasoning_tokens = _extract_reasoning_tokens(usage)

        reasoning_present = (
            bool(message.get("reasoning_details"))
            or "<think>" in raw_text.lower()
            or (reasoning_tokens is not None and reasoning_tokens > 0)
        )
        if not text:
            if reasoning_present and finish_reason == "length":
                raise RuntimeError(
                    "LLM response exhausted the completion budget before emitting a final answer. "
                    "Increase generation.max_completion_tokens."
                )
            if reasoning_present:
                raise RuntimeError("LLM response contained reasoning but no final answer.")
            raise RuntimeError("LLM response did not contain a final answer.")

        return GenerationResult(
            text=text,
            input_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            output_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            reasoning_tokens=reasoning_tokens,
            cost_usd=self._compute_cost(
                prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens if isinstance(completion_tokens, int) else None,
            ),
            provider="openai_compatible",
            model=self.model,
        )
