from __future__ import annotations

from typing import TYPE_CHECKING

from src.generation.base import GenerationResult
from src.generation.openai_compatible import DEFAULT_SYSTEM_PROMPT, _format_contexts
from src.logging_utils import get_logger
from src.types import Document

if TYPE_CHECKING:
    import anthropic as _anthropic

logger = get_logger(__name__)


class AnthropicCompatibleGenerator:
    """Generator backed by the Anthropic Messages API (via the ``anthropic`` SDK).

    Intended for models that expose an Anthropic-format endpoint rather than
    an OpenAI-compatible one — e.g. minimax-m2.7 served behind an Anthropic
    proxy.  Configuration mirrors :class:`OpenAICompatibleGenerator` where
    applicable; Anthropic-specific parameters (``max_tokens``, token usage
    field names) are handled transparently.

    Args:
        model: Model identifier forwarded to the API.
        api_key: Bearer token for authentication.
        api_base: Optional base URL override (passed as ``base_url`` to
            :class:`anthropic.Anthropic`).  When ``None`` the SDK default is
            used (``https://api.anthropic.com``).
        system_prompt: System message text.
        temperature: Sampling temperature.
        max_output_tokens: Maximum tokens in the response (``max_tokens``).
        timeout_sec: Per-request timeout in seconds.
        input_price_per_1m: Cost per 1 M input tokens (USD).
        output_price_per_1m: Cost per 1 M output tokens (USD).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_output_tokens: int = 128,
        timeout_sec: int = 60,
        input_price_per_1m: float = 0.0,
        output_price_per_1m: float = 0.0,
    ) -> None:
        import anthropic

        self.model = model
        self.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be > 0, got {self.max_output_tokens}")
        self.timeout_sec = int(timeout_sec)
        self.input_price_per_1m = float(input_price_per_1m)
        self.output_price_per_1m = float(output_price_per_1m)

        client_kwargs: dict = {"api_key": api_key, "timeout": float(self.timeout_sec)}
        if api_base:
            client_kwargs["base_url"] = api_base.rstrip("/")
        self._client: _anthropic.Anthropic = anthropic.Anthropic(**client_kwargs)

    def _compute_cost(
        self, input_tokens: int | None, output_tokens: int | None
    ) -> float | None:
        if input_tokens is None or output_tokens is None:
            return None
        if self.input_price_per_1m <= 0.0 and self.output_price_per_1m <= 0.0:
            return None
        return (
            (input_tokens * self.input_price_per_1m)
            + (output_tokens * self.output_price_per_1m)
        ) / 1_000_000.0

    def generate(self, question: str, contexts: list[Document]) -> GenerationResult:
        if not contexts:
            return GenerationResult(
                text="I do not know.",
                provider="anthropic",
                model=self.model,
            )

        import anthropic

        user_prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Retrieved context:\n{_format_contexts(contexts)}\n\n"
            "Answer using the most relevant parts of the context above. "
            "Provide the best factual answer you can."
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.temperature,
            )
        except anthropic.APIStatusError as exc:
            logger.warning("Anthropic API status error %s: %.200s", exc.status_code, exc.message)
            raise RuntimeError(
                f"Anthropic request failed with status {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            logger.warning("Anthropic connection error: %s", exc)
            raise RuntimeError(f"Anthropic request failed: {exc}") from exc

        if not response.content:
            raise RuntimeError("Anthropic response did not contain any content blocks.")

        text = response.content[0].text.strip()
        if not text:
            raise RuntimeError("Anthropic response did not contain a final answer.")

        usage = response.usage
        input_tokens: int | None = getattr(usage, "input_tokens", None)
        output_tokens: int | None = getattr(usage, "output_tokens", None)

        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._compute_cost(input_tokens, output_tokens),
            provider="anthropic",
            model=self.model,
        )
